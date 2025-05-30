// Copyright 2018 The gVisor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package testutil contains utility functions for runsc tests.
package testutil

import (
	"bufio"
	"context"
	"debug/elf"
	"encoding/base32"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/cenkalti/backoff"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"golang.org/x/sys/unix"
	"gvisor.dev/gvisor/pkg/rand"
	"gvisor.dev/gvisor/pkg/sentry/watchdog"
	"gvisor.dev/gvisor/pkg/sync"
	"gvisor.dev/gvisor/runsc/config"
	"gvisor.dev/gvisor/runsc/flag"
	"gvisor.dev/gvisor/runsc/specutils"
)

var (
	partition       = flag.Int("partition", IntFromEnv("PARTITION", 1), "partition number, this is 1-indexed")
	totalPartitions = flag.Int("total_partitions", IntFromEnv("TOTAL_PARTITIONS", 1), "total number of partitions")
	runscPath       = flag.String("runsc", os.Getenv("RUNTIME"), "path to runsc binary")

	// Flags controlling features for sandbox under test, prefixed with
	// "test-" to avoid potential conflicts with runsc flags.
	checkpointSupported              = flag.Bool("test-checkpoint", BoolFromEnv("TEST_CHECKPOINT", true), "control checkpoint/restore support")
	isRunningWithOverlay             = flag.Bool("test-overlay", BoolFromEnv("TEST_OVERLAY", false), "whether test is running with --overlay2")
	isRunningWithNetRaw              = flag.Bool("test-net-raw", BoolFromEnv("TEST_NET_RAW", false), "whether test is running with raw socket support")
	isRunningWithHostNet             = flag.Bool("test-hostnet", BoolFromEnv("TEST_HOSTNET", false), "whether test is running with hostnet")
	isRunningWithSaveRestoreNetstack = flag.Bool("test-save-restore-netstack", BoolFromEnv("TEST_SAVE_RESTORE_NETSTACK", false), "whether test is running with --save-restore-netstack")
	// TestEnvSupportsNetAdmin indicates whether a test sandbox can perform
	// all net admin tasks. Note that some test environments cannot perform
	// some tasks despite the presence of CAP_NET_ADMIN.
	TestEnvSupportsNetAdmin = true
)

// StringFromEnv returns the value of the named environment variable, or `def` if unset/empty.
// It is useful for defining flags where the default value can be specified through the environment.
func StringFromEnv(name, def string) string {
	str := os.Getenv(name)
	if str == "" {
		return def
	}
	return str
}

// IntFromEnv returns the integer value of the named environment variable, or `def` if unset/empty.
// It is useful for defining flags where the default value can be specified through the environment.
func IntFromEnv(name string, def int) int {
	str := os.Getenv(name)
	if str == "" {
		return def
	}
	v, err := strconv.ParseInt(str, 10, 64)
	if err != nil {
		// N.B. This library is testonly, so a panic here is reasonable.
		panic(fmt.Errorf("invalid environment variable %q; got %q expected integer: %w", name, str, err))
	}
	return int(v)
}

// BoolFromEnv returns the boolean value of the named environment variable, or `def` if unset/empty.
// It is useful for defining flags where the default value can be specified through the environment.
func BoolFromEnv(name string, def bool) bool {
	str := strings.ToLower(os.Getenv(name))
	if str == "" {
		return def
	}
	v, err := strconv.ParseBool(str)
	if err != nil {
		panic(fmt.Errorf("invalid environment variable %q; got %q expected bool: %w", name, str, err))
	}
	return v
}

// DurationFromEnv returns the duration of the named environment variable, or `def` if unset/empty.
// It is useful for defining flags where the default value can be specified through the environment.
func DurationFromEnv(name string, def time.Duration) time.Duration {
	str := strings.ToLower(os.Getenv(name))
	if str == "" {
		return def
	}
	d, err := time.ParseDuration(str)
	if err != nil {
		panic(fmt.Errorf("invalid environment variable %q; got %q expected duration: %w", name, str, err))
	}
	return d
}

// IsCheckpointSupported returns the relevant command line flag.
func IsCheckpointSupported() bool {
	return *checkpointSupported
}

// IsRunningWithHostNet returns the relevant command line flag.
func IsRunningWithHostNet() bool {
	return *isRunningWithHostNet
}

// IsRunningWithNetRaw returns the relevant command line flag.
func IsRunningWithNetRaw() bool {
	return *isRunningWithNetRaw
}

// IsRunningWithOverlay returns the relevant command line flag.
func IsRunningWithOverlay() bool {
	return *isRunningWithOverlay
}

// IsRunningWithSaveRestoreNetstack returns the relevant command line flag.
func IsRunningWithSaveRestoreNetstack() bool {
	return *isRunningWithSaveRestoreNetstack
}

// ImageByName mangles the image name used locally. This depends on the image
// build infrastructure in images/ and tools/vm.
func ImageByName(name string) string {
	return fmt.Sprintf("gvisor.dev/images/%s", name)
}

// ConfigureExePath configures the executable for runsc in the test environment.
func ConfigureExePath() error {
	if *runscPath == "" {
		path, err := FindFile("runsc/runsc")
		if err != nil {
			return err
		}
		*runscPath = path
	}
	specutils.ExePath = *runscPath
	return nil
}

// TmpDir returns the absolute path to a writable directory that can be used as
// scratch by the test.
func TmpDir() string {
	if dir, ok := os.LookupEnv("TEST_TMPDIR"); ok {
		return dir
	}
	return "/tmp"
}

// Logger is a simple logging wrapper.
//
// This is designed to be implemented by *testing.T.
type Logger interface {
	Name() string
	Logf(fmt string, args ...any)
}

// DefaultLogger logs using the log package.
type DefaultLogger string

// Name implements Logger.Name.
func (d DefaultLogger) Name() string {
	return string(d)
}

// Logf implements Logger.Logf.
func (d DefaultLogger) Logf(fmt string, args ...any) {
	log.Printf(fmt, args...)
}

// multiLogger logs to multiple Loggers.
type multiLogger []Logger

// Name implements Logger.Name.
func (m multiLogger) Name() string {
	names := make([]string, len(m))
	for i, l := range m {
		names[i] = l.Name()
	}
	return strings.Join(names, "+")
}

// Logf implements Logger.Logf.
func (m multiLogger) Logf(fmt string, args ...any) {
	for _, l := range m {
		l.Logf(fmt, args...)
	}
}

// NewMultiLogger returns a new Logger that logs on multiple Loggers.
func NewMultiLogger(loggers ...Logger) Logger {
	return multiLogger(loggers)
}

// Cmd is a simple wrapper.
type Cmd struct {
	logger Logger
	*exec.Cmd
}

// CombinedOutput returns the output and logs.
func (c *Cmd) CombinedOutput() ([]byte, error) {
	out, err := c.Cmd.CombinedOutput()
	if len(out) > 0 {
		c.logger.Logf("output: %s", string(out))
	}
	if err != nil {
		c.logger.Logf("error: %v", err)
	}
	return out, err
}

// Command is a simple wrapper around exec.Command, that logs.
func Command(logger Logger, args ...string) *Cmd {
	logger.Logf("command: %s", strings.Join(args, " "))
	return &Cmd{
		logger: logger,
		Cmd:    exec.Command(args[0], args[1:]...),
	}
}

// TestConfig returns the default configuration to use in tests. Note that
// 'RootDir' must be set by caller if required.
func TestConfig(t *testing.T) *config.Config {
	logDir := os.TempDir()
	if dir, ok := os.LookupEnv("TEST_UNDECLARED_OUTPUTS_DIR"); ok {
		logDir = dir + "/"
	}

	testFlags := flag.NewFlagSet("test", flag.ContinueOnError)
	config.RegisterFlags(testFlags)
	conf, err := config.NewFromFlags(testFlags)
	if err != nil {
		t.Fatalf("error loading configuration from flags: %v", err)
	}
	// Change test defaults.
	conf.Debug = true
	conf.DebugLog = path.Join(logDir, "runsc.log."+t.Name()+".%TIMESTAMP%.%COMMAND%.txt")
	conf.LogPackets = true
	conf.Network = config.NetworkNone
	conf.Strace = true
	conf.TestOnlyAllowRunAsCurrentUserWithoutChroot = true
	conf.WatchdogAction = watchdog.Panic
	return conf
}

// NewSpecWithArgs creates a simple spec with the given args suitable for use
// in tests.
func NewSpecWithArgs(args ...string) *specs.Spec {
	return &specs.Spec{
		// The host filesystem root is the container root.
		Root: &specs.Root{
			Path:     "/",
			Readonly: true,
		},
		Process: &specs.Process{
			Args: args,
			Env: []string{
				"PATH=" + os.Getenv("PATH"),
			},
			Capabilities: specutils.AllCapabilities(),
			Cwd:          "/",
		},
		Mounts: []specs.Mount{
			// Hide the host /etc to avoid any side-effects.
			// For example, bash reads /etc/passwd and if it is
			// very big, tests can fail by timeout.
			{
				Type:        "tmpfs",
				Destination: "/etc",
			},
			// Root is readonly, but many tests want to write to tmpdir.
			// This creates a writable mount inside the root. Also, when tmpdir points
			// to "/tmp", it makes the actual /tmp to be mounted and not a tmpfs
			// inside the sentry.
			{
				Type:        "bind",
				Destination: TmpDir(),
				Source:      TmpDir(),
			},
		},
		Hostname: "runsc-test-hostname",
	}
}

// SetupRootDir creates a root directory for containers.
func SetupRootDir() (string, func(), error) {
	rootDir, err := os.MkdirTemp(TmpDir(), "containers")
	if err != nil {
		return "", nil, fmt.Errorf("error creating root dir: %v", err)
	}
	return rootDir, func() { os.RemoveAll(rootDir) }, nil
}

// SetupContainer creates a bundle and root dir for the container, generates a
// test config, and writes the spec to config.json in the bundle dir.
func SetupContainer(spec *specs.Spec, conf *config.Config) (rootDir, bundleDir string, cleanup func(), err error) {
	rootDir, rootCleanup, err := SetupRootDir()
	if err != nil {
		return "", "", nil, err
	}
	conf.RootDir = rootDir
	bundleDir, bundleCleanup, err := SetupBundleDir(spec)
	if err != nil {
		rootCleanup()
		return "", "", nil, err
	}
	return rootDir, bundleDir, func() {
		bundleCleanup()
		rootCleanup()
	}, err
}

// SetupBundleDir creates a bundle dir and writes the spec to config.json.
func SetupBundleDir(spec *specs.Spec) (string, func(), error) {
	bundleDir, err := os.MkdirTemp(TmpDir(), "bundle")
	if err != nil {
		return "", nil, fmt.Errorf("error creating bundle dir: %v", err)
	}
	cleanup := func() { os.RemoveAll(bundleDir) }
	if err := writeSpec(bundleDir, spec); err != nil {
		cleanup()
		return "", nil, fmt.Errorf("error writing spec: %v", err)
	}
	return bundleDir, cleanup, nil
}

// writeSpec writes the spec to disk in the given directory.
func writeSpec(dir string, spec *specs.Spec) error {
	b, err := json.Marshal(spec)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "config.json"), b, 0755)
}

// RandomID returns 20 random bytes following the given prefix.
func RandomID(prefix string) string {
	b := make([]byte, 20)
	// "[Read] always returns len(p) and a nil error." --godoc
	if _, err := rand.Read(b); err != nil {
		panic("rand.Read failed: " + err.Error())
	}
	if prefix != "" {
		prefix = prefix + "-"
	}
	return fmt.Sprintf("%s%s", prefix, base32.StdEncoding.EncodeToString(b))
}

// RandomContainerID generates a random container id for each test.
//
// The container id is used to create an abstract unix domain socket, which
// must be unique. While the container forbids creating two containers with the
// same name, sometimes between test runs the socket does not get cleaned up
// quickly enough, causing container creation to fail.
func RandomContainerID() string {
	return RandomID("test-container")
}

// Copy copies file from src to dst.
func Copy(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	st, err := in.Stat()
	if err != nil {
		return err
	}

	out, err := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, st.Mode().Perm())
	if err != nil {
		return err
	}
	defer out.Close()

	// Mirror the local user's permissions across all users. This is
	// because as we inject things into the container, the UID/GID will
	// change. Also, the build system may generate artifacts with different
	// modes. At the top-level (volume mapping) we have a big read-only
	// knob that can be applied to prevent modifications.
	//
	// Note that this must be done via a separate Chmod call, otherwise the
	// current process's umask will get in the way.
	var mode os.FileMode
	if st.Mode()&0100 != 0 {
		mode |= 0111
	}
	if st.Mode()&0200 != 0 {
		mode |= 0222
	}
	if st.Mode()&0400 != 0 {
		mode |= 0444
	}
	if err := os.Chmod(dst, mode); err != nil {
		return err
	}

	_, err = io.Copy(out, in)
	return err
}

// Poll is a shorthand function to poll for something with given timeout.
func Poll(cb func() error, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return PollContext(ctx, cb)
}

// PollContext is like Poll, but takes a context instead of a timeout.
func PollContext(ctx context.Context, cb func() error) error {
	b := backoff.WithContext(backoff.NewConstantBackOff(100*time.Millisecond), ctx)
	return backoff.Retry(cb, b)
}

// WaitForHTTP tries GET requests on a port until the call succeeds or timeout.
func WaitForHTTP(ip string, port int, timeout time.Duration) error {
	cb := func() error {
		c := &http.Client{
			// Calculate timeout to be able to do minimum 5 attempts.
			Timeout: timeout / 5,
		}
		url := fmt.Sprintf("http://%s:%d/", ip, port)
		resp, err := c.Get(url)
		if err != nil {
			log.Printf("Waiting %s: %v", url, err)
			return err
		}
		resp.Body.Close()
		return nil
	}
	return Poll(cb, timeout)
}

// Reaper reaps child processes.
type Reaper struct {
	// mu protects ch, which will be nil if the reaper is not running.
	mu sync.Mutex
	ch chan os.Signal
}

// Start starts reaping child processes.
func (r *Reaper) Start() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.ch != nil {
		panic("reaper.Start called on a running reaper")
	}

	r.ch = make(chan os.Signal, 1)
	signal.Notify(r.ch, unix.SIGCHLD)

	go func() {
		for {
			r.mu.Lock()
			ch := r.ch
			r.mu.Unlock()
			if ch == nil {
				return
			}

			_, ok := <-ch
			if !ok {
				// Channel closed.
				return
			}
			for {
				cpid, _ := unix.Wait4(-1, nil, unix.WNOHANG, nil)
				if cpid < 1 {
					break
				}
			}
		}
	}()
}

// Stop stops reaping child processes.
func (r *Reaper) Stop() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.ch == nil {
		panic("reaper.Stop called on a stopped reaper")
	}

	signal.Stop(r.ch)
	close(r.ch)
	r.ch = nil
}

// StartReaper is a helper that starts a new Reaper and returns a function to
// stop it.
func StartReaper() func() {
	r := &Reaper{}
	r.Start()
	return r.Stop
}

// WaitUntilRead reads from the given reader until the wanted string is found
// or until timeout.
func WaitUntilRead(r io.Reader, want string, timeout time.Duration) error {
	sc := bufio.NewScanner(r)
	// done must be accessed atomically. A value greater than 0 indicates
	// that the read loop can exit.
	doneCh := make(chan bool)
	defer close(doneCh)
	go func() {
		for sc.Scan() {
			t := sc.Text()
			if strings.Contains(t, want) {
				doneCh <- true
				return
			}
			select {
			case <-doneCh:
				return
			default:
			}
		}
		doneCh <- false
	}()

	select {
	case <-time.After(timeout):
		return fmt.Errorf("timeout waiting to read %q", want)
	case res := <-doneCh:
		if !res {
			return fmt.Errorf("reader closed while waiting to read %q", want)
		}
		return nil
	}
}

// KillCommand kills the process running cmd unless it hasn't been started. It
// returns an error if it cannot kill the process unless the reason is that the
// process has already exited.
//
// KillCommand will also reap the process.
func KillCommand(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	if err := cmd.Process.Kill(); err != nil {
		if !strings.Contains(err.Error(), "process already finished") {
			return fmt.Errorf("failed to kill process %v: %v", cmd, err)
		}
	}
	return cmd.Wait()
}

// WriteTmpFile writes text to a temporary file, closes the file, and returns
// the name of the file. A cleanup function is also returned.
func WriteTmpFile(pattern, text string) (string, func(), error) {
	file, err := os.CreateTemp(TmpDir(), pattern)
	if err != nil {
		return "", nil, err
	}
	defer file.Close()
	if _, err := file.Write([]byte(text)); err != nil {
		return "", nil, err
	}
	return file.Name(), func() { os.RemoveAll(file.Name()) }, nil
}

// IsStatic returns true iff the given file is a static binary.
func IsStatic(filename string) (bool, error) {
	f, err := elf.Open(filename)
	if err != nil {
		return false, err
	}
	for _, prog := range f.Progs {
		if prog.Type == elf.PT_INTERP {
			return false, nil // Has interpreter.
		}
	}
	return true, nil
}

// TouchShardStatusFile indicates to Bazel that the test runner supports
// sharding by creating or updating the last modified date of the file
// specified by TEST_SHARD_STATUS_FILE.
//
// See https://docs.bazel.build/versions/master/test-encyclopedia.html#role-of-the-test-runner.
func TouchShardStatusFile() error {
	if statusFile, ok := os.LookupEnv("TEST_SHARD_STATUS_FILE"); ok {
		cmd := exec.Command("touch", statusFile)
		if b, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("touch %q failed:\n output: %s\n error: %s", statusFile, string(b), err.Error())
		}
	}
	return nil
}

// TestIndicesForShard returns indices for this test shard based on the
// TEST_SHARD_INDEX and TEST_TOTAL_SHARDS environment vars, as well as
// the passed partition flags.
//
// If either of the env vars are not present, then the function will return all
// tests. If there are more shards than there are tests, then the returned list
// may be empty.
func TestIndicesForShard(numTests int) ([]int, error) {
	var (
		shardIndex = 0
		shardTotal = 1
	)

	indexStr, indexOk := os.LookupEnv("TEST_SHARD_INDEX")
	totalStr, totalOk := os.LookupEnv("TEST_TOTAL_SHARDS")
	if indexOk && totalOk {
		// Parse index and total to ints.
		var err error
		shardIndex, err = strconv.Atoi(indexStr)
		if err != nil {
			return nil, fmt.Errorf("invalid TEST_SHARD_INDEX %q: %v", indexStr, err)
		}
		shardTotal, err = strconv.Atoi(totalStr)
		if err != nil {
			return nil, fmt.Errorf("invalid TEST_TOTAL_SHARDS %q: %v", totalStr, err)
		}
	}

	// Combine with the partitions.
	partitionSize := shardTotal
	shardTotal = (*totalPartitions) * shardTotal
	shardIndex = partitionSize*(*partition-1) + shardIndex

	// Calculate!
	var indices []int
	numBlocks := int(math.Ceil(float64(numTests) / float64(shardTotal)))
	for i := 0; i < numBlocks; i++ {
		pick := i*shardTotal + shardIndex
		if pick < numTests {
			indices = append(indices, pick)
		}
	}
	return indices, nil
}
