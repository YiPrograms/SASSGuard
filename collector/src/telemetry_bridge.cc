#include "config.h"
#include "telemetry.h"

#include <pthread.h>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <string>

#include "debug.h"

namespace {

SGClientConfig telemetry_config{};
pthread_t telemetry_thread;
bool telemetry_initialized = false;
bool telemetry_started = false;

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

bool env_capture_disabled() {
  const char *value = getenv(SASSGUARD_CAPTURE_DISABLE_ENV);
  if (value == nullptr) return false;
  return strcmp(value, "1") == 0 || strcasecmp(value, "true") == 0 ||
         strcasecmp(value, "yes") == 0 || strcasecmp(value, "on") == 0;
}

void *start_go_client(void *) {
  sg_go_start(&telemetry_config);
  DEBUG("telemetry client started, server %s", telemetry_config.server_addr);
  return nullptr;
}

}  // namespace

__attribute__((constructor))
static void telemetry_constructor(void) {
  if (env_capture_disabled()) {
    DEBUG("telemetry disabled by %s", SASSGUARD_CAPTURE_DISABLE_ENV);
    return;
  }

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

  telemetry_initialized = sg_telemetry_init(telemetry_config.ring_capacity) != 0;
  if (!telemetry_initialized) {
    DEBUG("failed to initialize telemetry ring");
    return;
  }

  int err = pthread_create(&telemetry_thread, nullptr, start_go_client, nullptr);
  if (err == 0) {
    pthread_detach(telemetry_thread);
    telemetry_started = true;
  } else {
    DEBUG("failed to start telemetry client thread: %d", err);
  }
}

__attribute__((destructor))
static void telemetry_destructor(void) {
  if (telemetry_started) {
    sg_go_stop();
  }
  if (telemetry_initialized) {
    sg_telemetry_shutdown();
  }
}
