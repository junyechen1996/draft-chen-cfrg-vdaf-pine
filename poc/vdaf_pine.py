import sys
sys.path.append('draft-irtf-cfrg-vdaf/poc')

from vdaf import Vdaf, test_vdaf

class Pine(Vdaf):
    """The Pine VDAF."""
    pass

if __name__ == '__main__':
    # Expect this test to fail until Pine is implemented.
    test_vdaf(Pine, None, [], 23)
