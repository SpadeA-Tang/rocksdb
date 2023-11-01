#pragma once
#include "table/block_based/block_based_table_reader.h"
#include "table/block_based/block_prefetcher.h"
#include "table/block_based/reader_common.h"
#include "table/separated_block_based/separated_block_based_table_reader.h"
#include "table/separated_block_based/separated_block_based_table_reader_impl.h"

namespace ROCKSDB_NAMESPACE {
class SeparatedBlockBasedTableIterator : public InternalIteratorBase<Slice> {
 public:
  SeparatedBlockBasedTableIterator(
      const SeparatedBlockBasedTable* table, const ReadOptions& read_options,
      const InternalKeyComparator& icomp,
      std::unique_ptr<InternalIteratorBase<IndexValue>>&& index_iter,
      std::unique_ptr<InternalIteratorBase<IndexValue>>&& old_index_iter,
      bool check_filter, bool need_upper_bound_check,
      const SliceTransform* prefix_extractor, TableReaderCaller caller,
      size_t compaction_readahead_size = 0, bool allow_unprepared_value = false)
      : index_iter_(std::move(index_iter)),
        old_index_iter_(std::move(old_index_iter)),
        iter_state_(IterState::NewVersion),
        table_(table),
        read_options_(read_options),
        icomp_(icomp),
        user_comparator_(icomp.user_comparator()),
        pinned_iters_mgr_(nullptr),
        prefix_extractor_(prefix_extractor),
        lookup_context_(caller),
        block_prefetcher_(compaction_readahead_size),
        allow_unprepared_value_(allow_unprepared_value),
        block_iter_points_to_real_block_(false),
        need_upper_bound_check_(need_upper_bound_check) {}

  ~SeparatedBlockBasedTableIterator() {}

  void Seek(const Slice& target) override;
  void SeekForPrev(const Slice& target) override;
  void SeekToFirst() override;
  void SeekToLast() override;
  void Next() final override;
  bool NextAndGetResult(IterateResult* result) override;
  void Prev() override;

  void ResetDataIter() {
    if (block_iter_points_to_real_block_) {
      if (pinned_iters_mgr_ != nullptr && pinned_iters_mgr_->PinningEnabled()) {
        block_iter_.DelegateCleanupsTo(pinned_iters_mgr_);
      }
      block_iter_.Invalidate(Status::OK());
      block_iter_points_to_real_block_ = false;
    }
    block_upper_bound_check_ = BlockUpperBound::kUnknown;
  }

  void ResetOldDataIter() {
    if (old_block_iter_points_to_real_block_) {
      if (pinned_iters_mgr_ != nullptr && pinned_iters_mgr_->PinningEnabled()) {
        old_block_iter_.DelegateCleanupsTo(pinned_iters_mgr_);
      }
      old_block_iter_.Invalidate(Status::OK());
      old_block_iter_points_to_real_block_ = false;
    }
    old_block_upper_bound_check_ = BlockUpperBound::kUnknown;
  }

  bool Valid() const override {
    if (iter_state_ == IterState::NewVersion) {
      return !is_out_of_bound_ &&
             (is_at_first_key_from_index_ ||
              (block_iter_points_to_real_block_ && block_iter_.Valid()));
    } else {
      return !is_old_out_of_bound_ &&
             (is_old_at_first_key_from_index_ ||
              (old_block_iter_points_to_real_block_ && old_block_iter_.Valid()));
    }
  }

  Slice key() const override {
    assert(Valid());
    if (iter_state_ == IterState::NewVersion) {
      return key_impl();
    } else {
      return old_key_impl();
    }
  }

  Slice key_impl() const{
    assert(Valid());
    if (is_at_first_key_from_index_) {
      return index_iter_->value().first_internal_key;
    } else {
      return block_iter_.key();
    }
  }

  Slice old_key_impl() const{
    assert(Valid());
    if (is_old_at_first_key_from_index_) {
      return old_index_iter_->value().first_internal_key;
    } else {
      return old_block_iter_.key();
    }
  }

  Slice user_key() const override {
    if (iter_state_ == IterState::NewVersion) {
      return user_key_impl();
    } else {
      return user_old_key_impl();
    }
  }

  Slice user_key_impl() const {
    assert(Valid());
    if (is_at_first_key_from_index_) {
      return ExtractUserKey(index_iter_->value().first_internal_key);
    } else {
      return block_iter_.user_key();
    }
  }

  Slice user_old_key_impl() const {
    assert(Valid());
    if (is_old_at_first_key_from_index_) {
      return ExtractUserKey(old_index_iter_->value().first_internal_key);
    } else {
      return old_block_iter_.user_key();
    }
  }

  bool PrepareValue() override {
    assert(Valid());

    if (!is_at_first_key_from_index_) {
      return true;
    }

    return const_cast<SeparatedBlockBasedTableIterator*>(this)
        ->MaterializeCurrentBlock();
  }

  Slice value() const override {
    if (iter_state_ == IterState::NewVersion) {
      return value_impl();
    } else {
      return old_value_impl();
    }
  }

  Slice value_impl() const  {
    // PrepareValue() must have been called.
    assert(!is_at_first_key_from_index_);
    assert(Valid());

    return block_iter_.value();
  }

  Slice old_value_impl() const  {
    // PrepareValue() must have been called.
    assert(!is_old_at_first_key_from_index_);
    assert(Valid());

    return old_block_iter_.value();
  }

  Status status() const override {
    if (iter_state_ == IterState::NewVersion) {
      return status_impl();
    } else {
      return old_status_impl();
    }
  }

  Status status_impl() const {
    // Prefix index set status to NotFound when the prefix does not exist
    if (!index_iter_->status().ok() && !index_iter_->status().IsNotFound()) {
      return index_iter_->status();
    } else if (block_iter_points_to_real_block_) {
      return block_iter_.status();
    } else {
      return Status::OK();
    }
  }

  Status old_status_impl() const {
    // Prefix index set status to NotFound when the prefix does not exist
    if (!old_index_iter_->status().ok() && !old_index_iter_->status().IsNotFound()) {
      return old_index_iter_->status();
    } else if (old_block_iter_points_to_real_block_) {
      return old_block_iter_.status();
    } else {
      return Status::OK();
    }
  }

  std::unique_ptr<InternalIteratorBase<IndexValue>> index_iter_;
  std::unique_ptr<InternalIteratorBase<IndexValue>> old_index_iter_;

 private:
  enum class IterDirection {
    kForward,
    kBackward,
  };
  // This enum indicates whether the upper bound falls into current block
  // or beyond.
  //   +-------------+
  //   |  cur block  |       <-- (1)
  //   +-------------+
  //                         <-- (2)
  //  --- <boundary key> ---
  //                         <-- (3)
  //   +-------------+
  //   |  next block |       <-- (4)
  //        ......
  //
  // When the block is smaller than <boundary key>, kUpperBoundInCurBlock
  // is the value to use. The examples are (1) or (2) in the graph. It means
  // all keys in the next block or beyond will be out of bound. Keys within
  // the current block may or may not be out of bound.
  // When the block is larger or equal to <boundary key>,
  // kUpperBoundBeyondCurBlock is to be used. The examples are (3) and (4)
  // in the graph. It means that all keys in the current block is within the
  // upper bound and keys in the next block may or may not be within the uppder
  // bound.
  // If the boundary key hasn't been checked against the upper bound,
  // kUnknown can be used.
  enum class BlockUpperBound {
    kUpperBoundInCurBlock,
    kUpperBoundBeyondCurBlock,
    kUnknown,
  };

  enum class IterState {
    NewVersion,
    OldVersion,
    OldVersioDone,
  };
  IterState iter_state_;

  const SeparatedBlockBasedTable* table_;
  const ReadOptions& read_options_;
  const InternalKeyComparator& icomp_;
  UserComparatorWrapper user_comparator_;
  PinnedIteratorsManager* pinned_iters_mgr_;
  DataBlockIter block_iter_;
  DataBlockIter old_block_iter_;
  const SliceTransform* prefix_extractor_;
  uint64_t prev_block_offset_ = std::numeric_limits<uint64_t>::max();
  uint64_t prev_old_block_offset_ = std::numeric_limits<uint64_t>::max();
  BlockCacheLookupContext lookup_context_;

  BlockPrefetcher block_prefetcher_;

  const bool allow_unprepared_value_;
  // True if block_iter_ is initialized and points to the same block
  // as index iterator.
  bool block_iter_points_to_real_block_;
  bool old_block_iter_points_to_real_block_;
  // See InternalIteratorBase::IsOutOfBound().
  bool is_out_of_bound_ = false;
  bool is_old_out_of_bound_ = false;
  // How current data block's boundary key with the next block is compared with
  // iterate upper bound.
  BlockUpperBound block_upper_bound_check_ = BlockUpperBound::kUnknown;
  BlockUpperBound old_block_upper_bound_check_ = BlockUpperBound::kUnknown;
  // True if we're standing at the first key of a block, and we haven't loaded
  // that block yet. A call to PrepareValue() will trigger loading the block.
  bool is_at_first_key_from_index_ = false;
  bool is_old_at_first_key_from_index_ = false;
  bool check_filter_;
  bool need_upper_bound_check_;

  // If `target` is null, seek to first.
  void SeekImpl(const Slice* target);

  void InitDataBlock();
  void InitOldDataBlock();
  bool MaterializeCurrentBlock();
  void FindKeyForward();
  void FindOldKeyForward();
  void FindBlockForward();
  void FindOldBlockForward();
  void FindKeyBackward();
  void CheckOutOfBound();
  void CheckOldOutOfBound();

  bool CheckPrefixMayMatch(const Slice& ikey, IterDirection direction) {
    if (need_upper_bound_check_ && direction == IterDirection::kBackward) {
      // Upper bound check isn't sufficient for backward direction to
      // guarantee the same result as total order, so disable prefix
      // check.
      return true;
    }
    //    if (check_filter_ &&
    //        !table_->PrefixMayMatch(ikey, read_options_, prefix_extractor_,
    //                                need_upper_bound_check_,
    //                                &lookup_context_)) {
    //      // TODO remember the iterator is invalidated because of prefix
    //      // match. This can avoid the upper level file iterator to falsely
    //      // believe the position is the end of the SST file and move to
    //      // the first key of the next file.
    //      ResetDataIter();
    //      return false;
    //    }
    return true;
  }

  void CheckDataBlockWithinUpperBound();
  void CheckOldDataBlockWithinUpperBound();

  void ParseItem(const Slice* target = nullptr);

  uint64_t current_key_version() const;

  bool seek_to_version(uint64_t ts);
  bool next_version(uint64_t ts);
  bool same_old_key();
  void seek_old_block(uint64_t ts);
  void NextImpl();
};

}  // namespace ROCKSDB_NAMESPACE