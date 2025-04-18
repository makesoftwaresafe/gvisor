"""Sender script for tcpdump test.
"""
# Copyright 2025 The gVisor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import socket


def main():
  sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sk.bind(("127.0.0.1", 9999))
  sk.connect(("127.0.0.1", 9999))
  assert sk.send(b"test",) == 4
  assert sk.send(b"testtest",) == 8
  return 0

if __name__ == "__main__":
  exit(main())
