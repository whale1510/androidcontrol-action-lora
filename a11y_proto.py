"""AndroidControl's accessibility trees are serialized `android_env.AndroidAccessibilityForest` protos.

The message definitions are loaded from third_party/android_env/a11y_descriptors.binpb (a
FileDescriptorSet exported from android_env 1.2.3, Apache-2.0), so only the `protobuf` package is
needed, not android_env itself.
"""
import os

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

_DESC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "android_env",
                     "a11y_descriptors.binpb")
_cls = None


def forest_class():
    global _cls
    if _cls is None:
        fds = descriptor_pb2.FileDescriptorSet()
        with open(_DESC, "rb") as f:
            fds.ParseFromString(f.read())
        pool = descriptor_pool.DescriptorPool()
        for fd in fds.file:                                  # stored dependencies-first
            pool.AddSerializedFile(fd.SerializeToString())
        desc = pool.FindMessageTypeByName("android_env.AndroidAccessibilityForest")
        if hasattr(message_factory, "GetMessageClass"):      # protobuf >= 4.22
            _cls = message_factory.GetMessageClass(desc)
        else:
            _cls = message_factory.MessageFactory(pool).GetPrototype(desc)
    return _cls


def load_forest(path):
    forest = forest_class()()
    with open(path, "rb") as f:
        forest.ParseFromString(f.read())
    return forest
