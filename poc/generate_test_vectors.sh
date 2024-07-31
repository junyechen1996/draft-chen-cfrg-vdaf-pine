version=`cat VERSION`
echo "PINE version: ${version}"
TEST_VECTOR=TRUE TEST_VECTOR_PATH=test_vec/$(printf "%02d" $version) sage -python -m unittest tests/test_vdaf.py
