#pip install cython numpy
python3 setup.py build_ext --inplace
# then verify:
python3 warlok_gw/tests/test_gateway.py
