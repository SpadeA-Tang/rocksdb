#include "table/separated_block_based/separated_block_based_table_builder.h"

#include <assert.h>
#include <stdio.h>

#include <atomic>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>

#include "rocksdb/cache.h"
#include "rocksdb/comparator.h"
#include "rocksdb/env.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/flush_block_policy.h"
#include "rocksdb/merge_operator.h"
#include "rocksdb/table.h"
#include "rocksdb/types.h"
#include "table/block_based/partitioned_filter_block.h"
#include "table/format.h"
#include "table/meta_blocks.h"
#include "table/table_builder.h"
#include "util/coding.h"
#include "util/compression.h"
#include "util/stop_watch.h"
#include "util/string_util.h"
#include "util/work_queue.h"

namespace ROCKSDB_NAMESPACE {

void SeparatedBlockBasedTableBuilder::Add(const Slice& key,
                                          const Slice& value) {
  Rep* r = rep_;
  assert(rep_->state != Rep::State::kClosed);
  if (!ok()) return;
  ValueType value_type = ExtractValueType(key);
  if (IsValueType(value_type)) {
#ifndef NDEBUG
  if (r->props.num_entries > r->props.num_range_deletions) {
    assert(r->internal_comparator.Compare(key, Slice(r->last_key)) > 0);
  }
#endif  // !NDEBUG
  } else if (value_type == kTypeRangeDeletion) {

  } else {
    assert(false);
  }

  r->props.num_entries++;
  r->props.raw_key_size += key.size();
  r->props.raw_value_size += value.size();
  if (value_type == kTypeDeletion || value_type == kTypeSingleDeletion) {
    r->props.num_deletions++;
  } else if (value_type == kTypeRangeDeletion) {
    r->props.num_deletions++;
    r->props.num_range_deletions++;
  } else if (value_type == kTypeMerge) {
    r->props.num_merge_operands++;
  }
}

struct SeparatedBlockBasedTableBuilder::Rep {
  const ImmutableOptions ioptions;
  const MutableCFOptions moptions;
  const BlockBasedTableOptions table_options;
  const InternalKeyComparator& internal_comparator;
  WritableFileWriter* file;
  std::atomic<uint64_t> offset;
  size_t alignment;
  BlockBuilder data_block;

  TableProperties props;
  
  enum class State {
    kBuffered,
    kUnbuffered,
    kClosed,
  };
  State state;
};

}  // namespace ROCKSDB_NAMESPACE