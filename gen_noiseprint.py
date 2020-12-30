import socialdetector.tf_options_setter

from data_loader import ucid_public, iplab_three
from socialdetector.dataset_utils import *
from socialdetector.experiment import DatasetSpec

from socialdetector.utility import log, jpeg_qtableinv, imread2f_pil


def generate_noiseprint_for_dataset(spec: DatasetSpec):
    for ds, label in spec.path_datasets:
        np_files = ds.map(path_bind(spec.noiseprint_path, spec.origin_path)).map(path_append(".npz"))
        entries_iterator = Dataset.zip((ds, np_files))
        generate_noiseprint_for_files(entries_iterator)


def generate_noiseprint_for_files(entries):
    from socialdetector.noiseprint import NoiseprintEngineV2

    engine = NoiseprintEngineV2()
    for entry in entries:
        src_file, dst_file = entry
        src_file = src_file.numpy().decode("utf-8")
        dst_file = dst_file.numpy().decode("utf-8")
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
        noiseprint = engine.predict_graphed(image)
        np.savez(dst_file, noiseprint.numpy().astype(np.float16))
        log("Saved noiseprint: %s" % dst_file)


generate_noiseprint_for_dataset(iplab_three)
