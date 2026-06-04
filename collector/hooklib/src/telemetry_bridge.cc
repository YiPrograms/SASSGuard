#include "config.h"
#include "telemetry.h"

#include <pthread.h>
#include <cstdlib>
#include <string>

#include "debug.h"

namespace {

SGClientConfig telemetry_config{};
pthread_t telemetry_thread;

uint32_t env_u32(const char *name, uint32_t fallback) {
#ifdef CUHOOK_DEBUG
  const char *value = getenv(name);
  if (value == nullptr || value[0] == '\0') return fallback;
  char *end = nullptr;
  unsigned long parsed = strtoul(value, &end, 10);
  if (end == value || parsed == 0) return fallback;
  return static_cast<uint32_t>(parsed);
#else
  (void)name;
  return fallback;
#endif
}

const char *env_string(const char *name, const char *fallback) {
#ifdef CUHOOK_DEBUG
  const char *value = getenv(name);
  if (value != nullptr && value[0] != '\0') return value;
#else
  (void)name;
#endif
  return fallback;
}

void *start_go_client(void *) {
  sg_go_start(&telemetry_config);
  DEBUG("telemetry client started, server %s", telemetry_config.server_addr);
  return nullptr;
}

}  // namespace

__attribute__((constructor))
static void telemetry_constructor(void) {
  telemetry_config.server_addr =
      env_string("SASSGUARD_SERVER_ADDR", SASSGUARD_DEFAULT_SERVER_ADDR);
  telemetry_config.ring_capacity =
      env_u32("SASSGUARD_RING_CAPACITY", SASSGUARD_DEFAULT_RING_CAPACITY);
  telemetry_config.drain_batch_size =
      env_u32("SASSGUARD_DRAIN_BATCH_SIZE",
              SASSGUARD_DEFAULT_DRAIN_BATCH_SIZE);
  telemetry_config.reconnect_backoff_ms =
      env_u32("SASSGUARD_RECONNECT_BACKOFF_MS",
              SASSGUARD_DEFAULT_RECONNECT_BACKOFF_MS);
#ifdef CUHOOK_DEBUG
  telemetry_config.debug_env_enabled = 1;
#else
  telemetry_config.debug_env_enabled = 0;
#endif
  telemetry_config.hook_version = SASSGUARD_HOOK_VERSION;

  sg_telemetry_init(telemetry_config.ring_capacity);

  int err = pthread_create(&telemetry_thread, nullptr, start_go_client, nullptr);
  if (err == 0) {
    pthread_detach(telemetry_thread);
  } else {
    DEBUG("failed to start telemetry client thread: %d", err);
  }
}

__attribute__((destructor))
static void telemetry_destructor(void) {
  sg_go_stop();
  sg_telemetry_shutdown();
}
