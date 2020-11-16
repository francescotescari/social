import os
import numpy as np
from typing import Callable

from socialdetector.noiseprint import gen_noiseprint, NoiseprintEngine
from socialdetector.utility import log, imread2f_pil, jpeg_qtableinv


class FilePathsLocations:
    files = []
    origin_dir = None

    def relative_paths(self):
        if self.origin_dir is None:
            return self.files
        return [filepath.replace(self.origin_dir, "") for filepath in self.files]

    @staticmethod
    def load_from(source_dir: str, extension: [str] = None):
        valid_extensions = None if extension is None else [x.lower() for x in extension]
        res = FilePathsLocations()
        res.origin_dir = source_dir
        filenames = res.files
        for file in os.listdir(source_dir):
            file = os.path.join(source_dir, file)
            if os.path.isfile(file) and (
                    valid_extensions is None or file.lower().endswith(tuple(valid_extensions))):
                filenames.append(file)
        return res




class NoiseprintDataGenerator:

    @staticmethod
    def generate_and_save(origin_path: str, destination_path: str, filter_fn: Callable):
        with NoiseprintEngine() as engine:
            for root, dirs, files in os.walk(origin_path, topdown=False):
                for name in files:
                    filename = os.path.join(root, name)
                    if not filter_fn(filename):
                        continue

                    output_filename = os.path.join(destination_path, "."+filename[len(origin_path):])
                    output_filename += ".npz"
                    if os.path.isfile(output_filename):
                        log("Skipping present file: %s" % output_filename)
                        continue
                    output_dir = output_filename[:-len(name)]
                    try:
                        os.makedirs(output_dir)
                    except FileExistsError:
                        pass

                    quality = jpeg_qtableinv(filename)
                    image = imread2f_pil(filename, channel=1)[0]
                    engine.load_session(quality)
                    noiseprint = engine.predict(image)
                    np.savez(output_filename, noiseprint.astype(np.float16))
                    log("Saved noiseprint: %s" % output_filename)


