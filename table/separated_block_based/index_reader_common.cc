//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
#include "table/separated_block_based/index_reader_common.h"

#include <iostream>

#include "table/separated_block_based/separated_block_based_table_reader.h"

namespace ROCKSDB_NAMESPACE {

Status SeparatedBlockBasedTable::IndexReaderCommon::ReadIndexBlock(
    const SeparatedBlockBasedTable* table, FilePrefetchBuffer* prefetch_buffer,
    const ReadOptions& read_options, bool use_cache, GetContext* get_context,
    BlockCacheLookupContext* lookup_context, CachableEntry<Block>* index_block,
    bool old_data_block) {
  PERF_TIMER_GUARD(read_index_block_nanos);

  assert(table != nullptr);
  assert(index_block != nullptr);
  assert(index_block->IsEmpty());

  const SeparatedBlockBasedTable::Rep* const rep = table->get_rep();
  assert(rep != nullptr);

  Status s = table->RetrieveBlock(
      prefetch_buffer, read_options,
      old_data_block ? rep->footer.old_index_handle()
                     : rep->footer.index_handle(),
      UncompressionDict::GetEmptyDict(), index_block, BlockType::kIndex,
      get_context, lookup_context, /* for_compaction */ false, use_cache,
      /* wait_for_cache */ true);

  return s;
}

Status SeparatedBlockBasedTable::IndexReaderCommon::GetOrReadIndexBlock(
    bool no_io, GetContext* get_context,
    BlockCacheLookupContext* lookup_context, CachableEntry<Block>* index_block,
    bool old_data_block) const {
  assert(index_block != nullptr);

  if (!index_block_.IsEmpty()) {
    index_block->SetUnownedValue(index_block_.GetValue());
    std::cout << "GetOrReadIndexBlock, data: "
              << Slice(index_block_.GetValue()->data(),
                       index_block_.GetValue()->size())
                     .ToString(true)
              << std::endl;
    return Status::OK();
  }

  ReadOptions read_options;
  if (no_io) {
    read_options.read_tier = kBlockCacheTier;
  }

  return ReadIndexBlock(table_, /*prefetch_buffer=*/nullptr, read_options,
                        cache_index_blocks(), get_context, lookup_context,
                        index_block, old_data_block);
}
}