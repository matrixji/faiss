/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/blockselect/BlockSelectImpl.cuh>

namespace faiss {
namespace gpu {

#ifndef __HIP_PLATFORM_HCC__
BLOCK_SELECT_IMPL(float, true, 32, 2);
BLOCK_SELECT_IMPL(float, false, 32, 2);
#endif

} // namespace gpu
} // namespace faiss
