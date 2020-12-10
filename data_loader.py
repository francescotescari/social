import socialdetector.tf_options_setter
from socialdetector.dataset_generator import my_data_generator, encode_coefficients_paper

seed = 1234321


def dataset_located(class_dirs, noiseprint_dir, origin_dir, default_validation, default_test):
    def get(batch=(256, 256, 0), encoding=None, noiseprint=False, **additional_configs):
        config = {
            'class_dirs': class_dirs,
            'seed': seed,
            'validate': default_validation,
            'test': default_test,
            'batch_size': batch,
            'dct_encoding': encoding,
            'shuffle': 5000
        }

        if noiseprint:
            config['noiseprint_path'] = noiseprint_dir
            config['origin_path'] = origin_dir

        config.update(additional_configs)

        train, validation, test = my_data_generator(**config)
        validation = validation.cache()
        test = test.cache()
        train = train

        return train, validation, test

    return get


ucid_social = dataset_located(
    noiseprint_dir="C:\\Users\\franz\\Downloads\\Datasets\\ucid_social\\ucid_noiseprint",
    origin_dir="C:\\Users\\franz\\Downloads\\Datasets\\ucid_social\\ucid",
    class_dirs=(
        "C:\\Users\\franz\\Downloads\\Datasets\\ucid_social\\ucid\\facebook\\**\\*.jpg",
        "C:\\Users\\franz\\Downloads\\Datasets\\ucid_social\\ucid\\flickr\\**\\*.jpg",
        "C:\\Users\\franz\\Downloads\\Datasets\\ucid_social\\ucid\\twitter\\**\\*.jpg"
    ),
    default_test=1000,
    default_validation=1000
)
