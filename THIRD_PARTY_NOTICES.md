# Third-party notices

This repository's own code is licensed under Apache-2.0 (see [LICENSE](LICENSE)).
It vendors the following third-party components under `runtime-library/deps/include/`.

| Component | Version | License | Copyright |
|---|---|---|---|
| [spdlog](https://github.com/gabime/spdlog) | 1.17.0 | MIT | Copyright (c) 2015-present, Gabi Melman & spdlog contributors |
| [{fmt}](https://github.com/fmtlib/fmt) (bundled with spdlog) | 12.1.0 | MIT | Copyright (c) 2012-present, Victor Zverovich & {fmt} contributors |
| [moodycamel::ConcurrentQueue](https://github.com/cameron314/concurrentqueue) | (single header) | BSD-2-Clause (also Boost-licensed) | Copyright (c) 2013-2020, Cameron Desrochers |

The full {fmt} license is kept next to its sources
(`spdlog/fmt/bundled/fmt.license.rst`) and the ConcurrentQueue license text is in the
header of `concurrentqueue.h`.

## MIT License (spdlog, {fmt})

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Not redistributed

These are used at build/run time but are **not** part of this repository and keep
their own licenses:

- **DX-RT** (DEEPX runtime, linked dynamically) and **DX-COM** (DEEPX compiler,
  pip-installed into the toolchain image from `sdk.deepx.ai`) — proprietary DEEPX software.
- DEEPX Model Zoo ONNX models downloaded by the test suite — see the license of each model.
