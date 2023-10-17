#include "db/table_properties_collector.h"
#include "file/file_util.h"
#include "options/options_helper.h"
#include "port/port.h"
#include "port/stack_trace.h"
#include "rocksdb/db.h"
#include "rocksdb/file_system.h"
#include "table/block_based/block_based_table_factory.h"
#include "table/block_based/partitioned_index_iterator.h"
#include "table/format.h"
#include "table/separated_block_based/separated_block_based_reader.h"
#include "table/separated_block_based/separated_block_based_table_builder.h"
#include "test_util/testharness.h"
#include "test_util/testutil.h"
#include "util/random.h"

namespace ROCKSDB_NAMESPACE {

class SeparatedBlockTest : public testing::Test {
 protected:
  SeparatedBlockTest() {}

  void CreateTable(const std::string& table_name,
                   const CompressionType& compression_type,
                   const std::map<std::string, std::string>& kv) {
    std::unique_ptr<WritableFileWriter> writer;
    NewFileWriter(table_name, &writer);

    // Create table builder.
    Options options;
    ImmutableOptions ioptions(options);
    InternalKeyComparator comparator(options.comparator);
    ColumnFamilyOptions cf_options;
    MutableCFOptions moptions(cf_options);
    IntTblPropCollectorFactories factories;
    SeparatedBlockBasedTableBuilder builder(
        BlockBasedTableOptions{},
        TableBuilderOptions(ioptions, moptions, comparator, &factories,
                            compression_type, CompressionOptions(),
                            0 /* column_family_id */, kDefaultColumnFamilyName,
                            -1 /* level */),
        BytewiseComparator(), writer.get());
    std::unique_ptr<TableBuilder> table_builder(builder);
  }

  std::string Path(const std::string& fname) { return test_dir_ + "/" + fname; }

 private:
  std::string test_dir_;
  std::shared_ptr<FileSystem> fs_;

  void NewFileWriter(const std::string& filename,
                     std::unique_ptr<WritableFileWriter>* writer) {
    std::string path = Path(filename);
    EnvOptions env_options;
    FileOptions foptions;
    std::unique_ptr<FSWritableFile> file;
    ASSERT_OK(fs_->NewWritableFile(path, foptions, &file, nullptr));
    writer->reset(new WritableFileWriter(std::move(file), path, env_options));
  }
};

TEST_F(SeparatedBlockTest, EmptyBuilder) {}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}