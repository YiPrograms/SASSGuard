package main

import (
	"encoding/json"
	"net"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadOnlineConfigDefaults(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "detection.json")
	raw := map[string]any{
		"enabled": false,
		"storage": map[string]any{},
	}
	data, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := loadOnlineConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Collector.ListenAddr != "127.0.0.1:59400" {
		t.Fatalf("unexpected listen addr %q", cfg.Collector.ListenAddr)
	}
	if cfg.Storage.CollectorOutputDir != "sassguard-data" {
		t.Fatalf("unexpected output dir %q", cfg.Storage.CollectorOutputDir)
	}
	if cfg.LaunchBatching.MaxBatchCount != 128 {
		t.Fatalf("unexpected batch count %d", cfg.LaunchBatching.MaxBatchCount)
	}
}

func TestJSONFrameRoundTrip(t *testing.T) {
	left, right := net.Pipe()
	defer left.Close()
	defer right.Close()

	want := map[string]any{"type": "kernel_launch_batch", "session_id": "abc"}
	errCh := make(chan error, 1)
	go func() {
		errCh <- writeJSONFrame(left, want, 1000)
	}()
	got, err := readJSONFrame(right, 1024, 1000)
	if err != nil {
		t.Fatal(err)
	}
	if err := <-errCh; err != nil {
		t.Fatal(err)
	}
	if got["type"] != want["type"] || got["session_id"] != want["session_id"] {
		t.Fatalf("unexpected frame: %#v", got)
	}
}

func TestLaunchBatcherFlushByCount(t *testing.T) {
	client := &onlineClient{sendCh: make(chan map[string]any, 1)}
	cfg := onlineConfig{}
	cfg.LaunchBatching.MaxBatchCount = 2
	cfg.LaunchBatching.MaxUnsentPerSession = 8
	cfg.LaunchBatching.FlushIntervalMs = 1000
	batcher := newLaunchBatcher(client, cfg)
	defer batcher.stop()

	batcher.add("s1", map[string]any{"sequence": 1})
	batcher.add("s1", map[string]any{"sequence": 2})
	message := <-client.sendCh
	if message["type"] != "kernel_launch_batch" {
		t.Fatalf("unexpected message type %v", message["type"])
	}
	launches, ok := message["launches"].([]map[string]any)
	if !ok || len(launches) != 2 {
		t.Fatalf("unexpected launches %#v", message["launches"])
	}
}
