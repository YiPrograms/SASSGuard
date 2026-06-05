package main

import (
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"

	"sassguard/hookgo/protocol"

	"google.golang.org/protobuf/proto"
)

type collector struct {
	root string
	mu   sync.Mutex
}

func main() {
	listenAddr := flag.String("listen", "127.0.0.1:59400", "TCP listen address")
	outDir := flag.String("out", "sassguard-data", "output directory")
	flag.Parse()

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fatal(err)
	}

	ln, err := net.Listen("tcp", *listenAddr)
	if err != nil {
		fatal(err)
	}
	defer ln.Close()

	c := &collector{root: *outDir}
	fmt.Fprintf(os.Stderr, "sassguard collector listening on %s, output %s\n", *listenAddr, *outDir)

	for {
		conn, err := ln.Accept()
		if err != nil {
			fatal(err)
		}
		go c.handleConn(conn)
	}
}

func (c *collector) handleConn(conn net.Conn) {
	defer conn.Close()

	for {
		payload, err := readFrame(conn)
		if err != nil {
			if err != io.EOF && err != io.ErrUnexpectedEOF {
				fmt.Fprintf(os.Stderr, "connection error: %v\n", err)
			}
			return
		}

		var env protocol.Envelope
		if err := proto.Unmarshal(payload, &env); err != nil {
			fmt.Fprintf(os.Stderr, "decode envelope: %v\n", err)
			return
		}
		if err := c.store(&env); err != nil {
			fmt.Fprintf(os.Stderr, "store event: %v\n", err)
			return
		}
	}
}

func readFrame(r io.Reader) ([]byte, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(hdr[:])
	buf := make([]byte, n)
	_, err := io.ReadFull(r, buf)
	return buf, err
}

func (c *collector) store(env *protocol.Envelope) error {
	if env.SessionId == "" {
		return fmt.Errorf("missing session_id")
	}

	sessionDir := filepath.Join(c.root, safeName(env.SessionId))

	c.mu.Lock()
	defer c.mu.Unlock()

	if err := os.MkdirAll(filepath.Join(sessionDir, "code"), 0o755); err != nil {
		return err
	}

	switch event := env.Event.(type) {
	case *protocol.Envelope_ProcessInfo:
		return c.storeProcessInfo(sessionDir, env, event.ProcessInfo)
	case *protocol.Envelope_ClientHello:
		return c.appendJSONL(sessionDir, "events.jsonl", map[string]any{
			"type":         "client_hello",
			"received_at":  time.Now().Format(time.RFC3339Nano),
			"session_id":   env.SessionId,
			"sequence":     env.Sequence,
			"timestamp_ns": env.TimestampNs,
			"pid":          env.Pid,
			"hello_pid":    event.ClientHello.Pid,
		})
	case *protocol.Envelope_Code:
		return c.storeCode(sessionDir, env, event.Code)
	case *protocol.Envelope_KernelLaunch:
		return c.appendJSONL(sessionDir, "events.jsonl", map[string]any{
			"type":              "kernel_launch",
			"received_at":       time.Now().Format(time.RFC3339Nano),
			"session_id":        env.SessionId,
			"sequence":          env.Sequence,
			"timestamp_ns":      env.TimestampNs,
			"pid":               env.Pid,
			"tid":               env.Tid,
			"kernel_name":       event.KernelLaunch.KernelName,
			"code_id":           event.KernelLaunch.CodeId,
			"grid_dim":          []uint32{event.KernelLaunch.GridDimX, event.KernelLaunch.GridDimY, event.KernelLaunch.GridDimZ},
			"block_dim":         []uint32{event.KernelLaunch.BlockDimX, event.KernelLaunch.BlockDimY, event.KernelLaunch.BlockDimZ},
			"shared_mem_bytes":  event.KernelLaunch.SharedMemBytes,
			"stream":            event.KernelLaunch.Stream,
			"device_pci_bus_id": event.KernelLaunch.DevicePciBusId,
		})
	case *protocol.Envelope_Stats:
		return c.appendJSONL(sessionDir, "events.jsonl", map[string]any{
			"type":           "stats",
			"received_at":    time.Now().Format(time.RFC3339Nano),
			"session_id":     env.SessionId,
			"sequence":       env.Sequence,
			"timestamp_ns":   env.TimestampNs,
			"pid":            env.Pid,
			"dropped_events": event.Stats.DroppedEvents,
			"dropped_bytes":  event.Stats.DroppedBytes,
			"queued_events":  event.Stats.QueuedEvents,
		})
	default:
		return c.appendJSONL(sessionDir, "events.jsonl", map[string]any{
			"type":         "unknown",
			"received_at":  time.Now().Format(time.RFC3339Nano),
			"session_id":   env.SessionId,
			"sequence":     env.Sequence,
			"timestamp_ns": env.TimestampNs,
			"pid":          env.Pid,
			"tid":          env.Tid,
		})
	}
}

func (c *collector) storeProcessInfo(sessionDir string, env *protocol.Envelope, info *protocol.ProcessInfoEvent) error {
	path := filepath.Join(sessionDir, "process.json")
	if _, err := os.Stat(path); err == nil {
		return c.appendJSONL(sessionDir, "events.jsonl", map[string]any{
			"type":         "process_info_duplicate",
			"received_at":  time.Now().Format(time.RFC3339Nano),
			"session_id":   env.SessionId,
			"sequence":     env.Sequence,
			"timestamp_ns": env.TimestampNs,
			"pid":          env.Pid,
		})
	}

	doc := map[string]any{
		"session_id":            env.SessionId,
		"received_at":           time.Now().Format(time.RFC3339Nano),
		"pid":                   env.Pid,
		"ppid":                  info.Ppid,
		"uid":                   info.Uid,
		"gid":                   info.Gid,
		"hostname":              info.Hostname,
		"exe_path":              info.ExePath,
		"cwd":                   info.Cwd,
		"argv":                  info.Argv,
		"process_start_time_ns": info.ProcessStartTimeNs,
		"hook_version":          info.HookVersion,
		"client_version":        info.ClientVersion,
	}
	data, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func (c *collector) storeCode(sessionDir string, env *protocol.Envelope, code *protocol.CodeEvent) error {
	hash := hex.EncodeToString(code.Sha256)
	if hash == "" {
		hash = "nohash"
	}
	name := fmt.Sprintf("code_%d_%s.bin", code.CodeId, hash)
	path := filepath.Join(sessionDir, "code", safeName(name))
	if err := os.WriteFile(path, code.Data, 0o644); err != nil {
		return err
	}
	return c.appendJSONL(sessionDir, "events.jsonl", map[string]any{
		"type":         "code",
		"received_at":  time.Now().Format(time.RFC3339Nano),
		"session_id":   env.SessionId,
		"sequence":     env.Sequence,
		"timestamp_ns": env.TimestampNs,
		"pid":          env.Pid,
		"tid":          env.Tid,
		"code_id":      code.CodeId,
		"code_type":    code.CodeType,
		"sha256":       hash,
		"size":         len(code.Data),
		"path":         filepath.ToSlash(filepath.Join("code", safeName(name))),
	})
}

func (c *collector) appendJSONL(sessionDir string, name string, value any) error {
	path := filepath.Join(sessionDir, name)
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer file.Close()

	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	_, err = file.Write(data)
	return err
}

func safeName(s string) string {
	out := []byte(s)
	for i, c := range out {
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-' {
			continue
		}
		out[i] = '_'
	}
	return string(out)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
