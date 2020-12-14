from tensorflow.python.data import Dataset
from tensorflow.python.data.experimental.ops.matching_files import MatchingFilesDataset

from data_loader import ucid_public
from socialdetector.dataset_generator import DatasetSpec
from socialdetector.dataset_utils import *

from socialdetector.utility import log, jpeg_qtableinv, imread2f_pil
import tensorflow.compat.v1 as tf1
import tensorflow as tf


def generate_noiseprint_for_dataset(spec: DatasetSpec):
    all_files = []
    for folder in spec.class_dirs:
        files = Dataset.list_files(folder, shuffle=False)
        np_files = files.map(path_bind(spec.noiseprint_path, spec.origin_path)).map(path_append(".npz"))
        entries_iterator = Dataset.zip((files, np_files))
        all_files.extend(list(entries_iterator.as_numpy_iterator()))

    tf1.disable_v2_behavior()
    tf.config.run_functions_eagerly(True)
    generate_noiseprint_for_files(all_files)


def generate_noiseprint_for_files(entries):
    from socialdetector.noiseprint import NoiseprintEngine

    with NoiseprintEngine() as engine:
        for entry in entries:
            src_file, dst_file = entry
            src_file = src_file.decode("utf-8")
            dst_file = dst_file.decode("utf-8")
            if os.path.isfile(dst_file):
                log("Skipping present file: %s" % dst_file)
                continue
            output_dir = os.path.split(os.path.abspath(dst_file))[0]
            try:
                print(output_dir)
                os.makedirs(output_dir)
            except FileExistsError:
                pass

            quality = jpeg_qtableinv(src_file)
            image = imread2f_pil(src_file, channel=1)[0]
            engine.load_session(quality)
            noiseprint = engine.predict(image)
            np.savez(dst_file, noiseprint.astype(np.float16))
            log("Saved noiseprint: %s" % dst_file)


generate_noiseprint_for_dataset(ucid_public)
