load("//tools:defs.bzl", "go_library")
load("//tools/go_generics:defs.bzl", "go_template_instance")

package(
    default_applicable_licenses = ["//:license"],
    licenses = ["notice"],
)

go_template_instance(
    name = "shm_refs",
    out = "shm_refs.go",
    consts = {
        "enableLogging": "true",
    },
    package = "shm",
    prefix = "Shm",
    template = "//pkg/refs:refs_template",
    types = {
        "T": "Shm",
    },
)

go_library(
    name = "shm",
    srcs = [
        "context.go",
        "shm.go",
        "shm_refs.go",
    ],
    visibility = ["//pkg/sentry:internal"],
    deps = [
        "//pkg/abi/linux",
        "//pkg/atomicbitops",
        "//pkg/context",
        "//pkg/errors/linuxerr",
        "//pkg/hostarch",
        "//pkg/log",
        "//pkg/refs",
        "//pkg/sentry/kernel/auth",
        "//pkg/sentry/kernel/ipc",
        "//pkg/sentry/ktime",
        "//pkg/sentry/memmap",
        "//pkg/sentry/pgalloc",
        "//pkg/sentry/usage",
        "//pkg/sentry/vfs",
        "//pkg/sync",
        "//pkg/usermem",
    ],
)
