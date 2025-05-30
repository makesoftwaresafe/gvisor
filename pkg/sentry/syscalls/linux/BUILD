load("//tools:defs.bzl", "go_library", "go_test")

package(
    default_applicable_licenses = ["//:license"],
    licenses = ["notice"],
)

go_library(
    name = "linux",
    srcs = [
        "error.go",
        "linux64.go",
        "path.go",
        "points.go",
        "sigset.go",
        "sys_afs_syscall.go",
        "sys_aio.go",
        "sys_capability.go",
        "sys_clone_amd64.go",
        "sys_clone_arm64.go",
        "sys_epoll.go",
        "sys_eventfd.go",
        "sys_file.go",
        "sys_futex.go",
        "sys_getdents.go",
        "sys_identity.go",
        "sys_inotify.go",
        "sys_iouring.go",
        "sys_key.go",
        "sys_membarrier.go",
        "sys_mempolicy.go",
        "sys_mmap.go",
        "sys_mount.go",
        "sys_mq.go",
        "sys_msgqueue.go",
        "sys_pipe.go",
        "sys_poll.go",
        "sys_prctl.go",
        "sys_process_vm.go",
        "sys_random.go",
        "sys_read_write.go",
        "sys_rlimit.go",
        "sys_rseq.go",
        "sys_rusage.go",
        "sys_sched.go",
        "sys_seccomp.go",
        "sys_sem.go",
        "sys_shm.go",
        "sys_signal.go",
        "sys_socket.go",
        "sys_splice.go",
        "sys_stat.go",
        "sys_stat_amd64.go",
        "sys_stat_arm64.go",
        "sys_sync.go",
        "sys_sysinfo.go",
        "sys_syslog.go",
        "sys_thread.go",
        "sys_time.go",
        "sys_timer.go",
        "sys_timerfd.go",
        "sys_tls_amd64.go",
        "sys_tls_arm64.go",
        "sys_utsname.go",
        "sys_xattr.go",
        "timespec.go",
    ],
    marshal = True,
    visibility = ["//:sandbox"],
    deps = [
        "//pkg/abi",
        "//pkg/abi/linux",
        "//pkg/atomicbitops",
        "//pkg/bits",
        "//pkg/bpf",
        "//pkg/context",
        "//pkg/errors/linuxerr",
        "//pkg/fspath",
        "//pkg/gohacks",
        "//pkg/hostarch",
        "//pkg/log",
        "//pkg/marshal",
        "//pkg/marshal/primitive",
        "//pkg/metric",
        "//pkg/rand",
        "//pkg/safemem",
        "//pkg/sentry/arch",
        "//pkg/sentry/fsimpl/eventfd",
        "//pkg/sentry/fsimpl/host",
        "//pkg/sentry/fsimpl/iouringfs",
        "//pkg/sentry/fsimpl/lock",
        "//pkg/sentry/fsimpl/pipefs",
        "//pkg/sentry/fsimpl/signalfd",
        "//pkg/sentry/fsimpl/timerfd",
        "//pkg/sentry/fsimpl/tmpfs",
        "//pkg/sentry/kernel",
        "//pkg/sentry/kernel/auth",
        "//pkg/sentry/kernel/fasync",
        "//pkg/sentry/kernel/ipc",
        "//pkg/sentry/kernel/mq",
        "//pkg/sentry/kernel/msgqueue",
        "//pkg/sentry/kernel/pipe",
        "//pkg/sentry/kernel/sched",
        "//pkg/sentry/kernel/shm",
        "//pkg/sentry/ktime",
        "//pkg/sentry/limits",
        "//pkg/sentry/loader",
        "//pkg/sentry/memmap",
        "//pkg/sentry/mm",
        "//pkg/sentry/seccheck",
        "//pkg/sentry/seccheck/points:points_go_proto",
        "//pkg/sentry/socket",
        "//pkg/sentry/socket/control",
        "//pkg/sentry/socket/unix/transport",
        "//pkg/sentry/syscalls",
        "//pkg/sentry/usage",
        "//pkg/sentry/vfs",
        "//pkg/sync",
        "//pkg/syserr",
        "//pkg/usermem",
        "//pkg/waiter",
        "@org_golang_google_protobuf//proto:go_default_library",
        "@org_golang_x_sys//unix:go_default_library",
    ],
)

go_test(
    name = "linux_test",
    size = "small",
    srcs = [
        "linux64_amd64_test.go",
        "linux64_arm64_test.go",
        "linux64_test.go",
    ],
    library = ":linux",
    deps = ["//pkg/sentry/seccheck"],
)
