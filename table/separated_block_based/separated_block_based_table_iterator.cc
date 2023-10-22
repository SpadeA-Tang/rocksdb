#include "table/separated_block_based/separated_block_based_table_iterator.h"

namespace ROCKSDB_NAMESPACE {
void SeparatedBlockBasedTableIterator::Seek(const Slice& target) {
  SeekImpl(&target);
}

void SeparatedBlockBasedTableIterator::SeekToFirst() { SeekImpl(nullptr); }

void SeparatedBlockBasedTableIterator::ParseItem() {
  SequenceNumber s(read_options_.snapshot->GetSequenceNumber());
  while (Valid()) {
    if (current_key_version() > s && !seek_to_version(s)) {
      Next();
      continue;
    }
    return;
  }
}

bool SeparatedBlockBasedTableIterator::seek_to_version(SequenceNumber s) {
  if (s >= current_key_version()) {
    return true;
  }

  while (next_version()) {
    if (s >= current_key_version()) {
      return true;
    }
  }

  return false;
}

SequenceNumber SeparatedBlockBasedTableIterator::current_key_version() const {
  auto k = key();
  uint64_t num = DecodeFixed64(k.data() + k.size() - kNumInternalBytes);
  unsigned char c = num & 0xff;
  return num >> 8;
}

bool SeparatedBlockBasedTableIterator::next_version() {
  if (iter_state_ == IterState::OldVersioDone) {
    return false;
  }
}

void SeparatedBlockBasedTableIterator::SeekImpl(const rocksdb::Slice* target) {
  is_out_of_bound_ = false;
  is_at_first_key_from_index_ = false;
  if (target && !CheckPrefixMayMatch(*target, IterDirection::kForward)) {
    ResetDataIter();
    return;
  }

  bool need_seek_index = true;
  if (block_iter_points_to_real_block_ && block_iter_.Valid()) {
  }

  if (need_seek_index) {
    if (target) {
      index_iter_->Seek(*target);
    } else {
      index_iter_->SeekToFirst();
    }

    if (!index_iter_->Valid()) {
      ResetDataIter();
    }
  }

  IndexValue v = index_iter_->value();
  const bool same_block = block_iter_points_to_real_block_ &&
                          v.handle.offset() == prev_block_offset_;

  if (!v.first_internal_key.empty() && !same_block &&
      (!target || icomp_.Compare(*target, v.first_internal_key) <= 0) &&
      allow_unprepared_value_) {
    // Index contains the first key of the block, and it's >= target.
    // We can defer reading the block.
    is_at_first_key_from_index_ = true;
    // ResetDataIter() will invalidate block_iter_. Thus, there is no need to
    // call CheckDataBlockWithinUpperBound() to check for iterate_upper_bound
    // as that will be done later when the data block is actually read.
    ResetDataIter();
  } else {
    // Need to use the data block.
    if (!same_block) {
      InitDataBlock();
    } else {
      // When the user does a reseek, the iterate_upper_bound might have
      // changed. CheckDataBlockWithinUpperBound() needs to be called
      // explicitly if the reseek ends up in the same data block.
      // If the reseek ends up in a different block, InitDataBlock() will do
      // the iterator upper bound check.
      CheckDataBlockWithinUpperBound();
    }

    if (target) {
      block_iter_.SeekWithoutTs(*target);
    } else {
      block_iter_.SeekToFirst();
    }
    FindKeyForward();
  }

  CheckOutOfBound();

  if (target) {
    assert(!Valid() || icomp_.Compare(*target, key()) <= 0);
  }
}

void SeparatedBlockBasedTableIterator::CheckDataBlockWithinUpperBound() {
  if (read_options_.iterate_upper_bound != nullptr &&
      block_iter_points_to_real_block_) {
    block_upper_bound_check_ = (user_comparator_.CompareWithoutTimestamp(
                                    *read_options_.iterate_upper_bound,
                                    /*a_has_ts=*/false, index_iter_->user_key(),
                                    /*b_has_ts=*/true) > 0)
                                   ? BlockUpperBound::kUpperBoundBeyondCurBlock
                                   : BlockUpperBound::kUpperBoundInCurBlock;
  }
}

void SeparatedBlockBasedTableIterator::FindKeyForward() {
  // This method's code is kept short to make it likely to be inlined.

  assert(!is_out_of_bound_);
  assert(block_iter_points_to_real_block_);

  if (!block_iter_.Valid()) {
    // This is the only call site of FindBlockForward(), but it's extracted into
    // a separate method to keep FindKeyForward() short and likely to be
    // inlined. When transitioning to a different block, we call
    // FindBlockForward(), which is much longer and is probably not inlined.
    FindBlockForward();
  } else {
    // This is the fast path that avoids a function call.
  }
}

void SeparatedBlockBasedTableIterator::FindBlockForward() {
  // TODO the while loop inherits from two-level-iterator. We don't know
  // whether a block can be empty so it can be replaced by an "if".
  do {
    if (!block_iter_.status().ok()) {
      return;
    }
    // Whether next data block is out of upper bound, if there is one.
    const bool next_block_is_out_of_bound =
        read_options_.iterate_upper_bound != nullptr &&
        block_iter_points_to_real_block_ &&
        block_upper_bound_check_ == BlockUpperBound::kUpperBoundInCurBlock;
    assert(!next_block_is_out_of_bound ||
           user_comparator_.CompareWithoutTimestamp(
               *read_options_.iterate_upper_bound, /*a_has_ts=*/false,
               index_iter_->user_key(), /*b_has_ts=*/true) <= 0);
    ResetDataIter();
    index_iter_->Next();
    if (next_block_is_out_of_bound) {
      // The next block is out of bound. No need to read it.
      TEST_SYNC_POINT_CALLBACK("BlockBasedTableIterator:out_of_bound", nullptr);
      // We need to make sure this is not the last data block before setting
      // is_out_of_bound_, since the index key for the last data block can be
      // larger than smallest key of the next file on the same level.
      if (index_iter_->Valid()) {
        is_out_of_bound_ = true;
      }
      return;
    }

    if (!index_iter_->Valid()) {
      return;
    }

    IndexValue v = index_iter_->value();

    if (!v.first_internal_key.empty() && allow_unprepared_value_) {
      // Index contains the first key of the block. Defer reading the block.
      is_at_first_key_from_index_ = true;
      return;
    }

    InitDataBlock();
    block_iter_.SeekToFirst();
  } while (!block_iter_.Valid());
}

void SeparatedBlockBasedTableIterator::CheckOutOfBound() {
  if (read_options_.iterate_upper_bound != nullptr &&
      block_upper_bound_check_ != BlockUpperBound::kUpperBoundBeyondCurBlock &&
      Valid()) {
    is_out_of_bound_ =
        user_comparator_.CompareWithoutTimestamp(
            *read_options_.iterate_upper_bound, /*a_has_ts=*/false, user_key(),
            /*b_has_ts=*/true) <= 0;
  }
}

void SeparatedBlockBasedTableIterator::SeekForPrev(const Slice& target) {}

void SeparatedBlockBasedTableIterator::SeekToLast() {}

void SeparatedBlockBasedTableIterator::Next() {
  if (is_at_first_key_from_index_ && !MaterializeCurrentBlock()) {
    return;
  }
  assert(block_iter_points_to_real_block_);
  block_iter_.Next();
  FindKeyForward();
  CheckOutOfBound();
}

bool SeparatedBlockBasedTableIterator::NextAndGetResult(IterateResult* result) {
  return InternalIteratorBase::NextAndGetResult(result);
}

void SeparatedBlockBasedTableIterator::Prev() {}

void SeparatedBlockBasedTableIterator::InitDataBlock() {
  BlockHandle data_block_handle = index_iter_->value().handle;
  if (!block_iter_points_to_real_block_ ||
      data_block_handle.offset() != prev_block_offset_ ||
      // if previous attempt of reading the block missed cache, try again
      block_iter_.status().IsIncomplete()) {
    if (block_iter_points_to_real_block_) {
      ResetDataIter();
    }
    auto* rep = table_->get_rep();
    bool is_for_compaction =
        lookup_context_.caller == TableReaderCaller::kCompaction;
    //    // Prefetch additional data for range scans (iterators).
    //    // Implicit auto readahead:
    //    //   Enabled after 2 sequential IOs when ReadOptions.readahead_size ==
    //    0.
    //    // Explicit user requested readahead:
    //    //   Enabled from the very first IO when ReadOptions.readahead_size is
    //    set. block_prefetcher_.PrefetchIfNeeded(rep, data_block_handle,
    //                                       read_options_.readahead_size,
    //                                       is_for_compaction);
    Status s;
    table_->NewDataBlockIterator<DataBlockIter>(
        read_options_, data_block_handle, &block_iter_, BlockType::kData,
        /*get_context=*/nullptr, &lookup_context_, s,
        block_prefetcher_.prefetch_buffer(),
        /*for_compaction=*/is_for_compaction);
    block_iter_points_to_real_block_ = true;
    CheckDataBlockWithinUpperBound();
  }
}

bool SeparatedBlockBasedTableIterator::MaterializeCurrentBlock() {
  return false;
}

void SeparatedBlockBasedTableIterator::FindKeyBackward() {}

}  // namespace ROCKSDB_NAMESPACE