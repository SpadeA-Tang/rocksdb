//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
#include "table/separated_block_based/index_reader_common.h"

#include "table/separated_block_based/separated_block_based_table_reader.h"

namespace ROCKSDB_NAMESPACE {

Status SeparatedBlockBasedTable::IndexReaderCommon::ReadIndexBlock(
    const SeparatedBlockBasedTable* table, FilePrefetchBuffer* prefetch_buffer,
    const ReadOptions& read_options, bool use_cache, GetContext* get_context,
    BlockCacheLookupContext* lookup_context, CachableEntry<Block>* index_block,
    CachableEntry<Block>* old_index_block) {
  PERF_TIMER_GUARD(read_index_block_nanos);

  assert(table != nullptr);
  assert(index_block != nullptr);
  assert(index_block->IsEmpty());
  assert(old_index_block != nullptr);
  assert(old_index_block->IsEmpty());

  const SeparatedBlockBasedTable::Rep* const rep = table->get_rep();
  assert(rep != nullptr);

  Status s = table->RetrieveBlock(
      prefetch_buffer, read_options, rep->footer.index_handle(),
      UncompressionDict::GetEmptyDict(), index_block, BlockType::kIndex,
      get_context, lookup_context, /* for_compaction */ false, use_cache,
      /* wait_for_cache */ true);

  if (!s.ok()) {
    return s;
  }

  s = table->RetrieveBlock(
      prefetch_buffer, read_options, rep->footer.old_index_handle(),
      UncompressionDict::GetEmptyDict(), old_index_block, BlockType::kIndex,
      get_context, lookup_context, /* for_compaction */ false, use_cache,
      /* wait_for_cache */ true);

  return s;
}

Status SeparatedBlockBasedTable::IndexReaderCommon::GetOrReadIndexBlock(
    bool no_io, GetContext* get_context,
    BlockCacheLookupContext* lookup_context,
    CachableEntry<Block>* index_block) const {
  assert(index_block != nullptr);

  if (!index_block_.IsEmpty()) {
    index_block->SetUnownedValue(index_block_.GetValue());
    return Status::OK();
  }

  ReadOptions read_options;
  if (no_io) {
    read_options.read_tier = kBlockCacheTier;
  }

  assert(false);
  return Status::OK();
//  return ReadIndexBlock(table_, /*prefetch_buffer=*/nullptr, read_options,
//                        cache_index_blocks(), get_context, lookup_context,
//                        index_block);
}

}