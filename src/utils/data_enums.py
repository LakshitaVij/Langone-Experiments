from enum import Enum

#INPUT_DIM = 224
#MEAN = 58.09
#STDDEV = 49.73


class SeriesType(Enum):
    AXT2 = {
        "crop_size": (300, 300),
        "pixel_range": (0, 1),
        "num_slices": 30,
        "key": "axt2",
    }
    ADC = {
        "crop_size": (180, 180),
        "pixel_range": (0, 1),
        "num_slices": 30,
        "key": "adc",
    }
    B1500 = {
        "crop_size": (180, 180),
        "pixel_range": (0, 1),
        "num_slices": 30,
        "key": "b1500",
    }
    DCE = {
        "crop_size": (180, 180),
        "pixel_range": (0, 1),
        "num_slices": 30,
        "key": "dce",
    }


def binarize_gleason_score(score):
    if score == 6:
        return 0
    elif score in [7, 8, 9, 10]:
        return 1
    else:
        return -1


def parse_string_to_pirads(pirads_string, max_pirads):
    # parse a string '[1, 2, 3]' to a list of ints [1, 2, 3]
    try:
        pirads_string = pirads_string.strip()
        pirads_string = pirads_string[1:-1]
        pirads_scores = [int(x.strip()) for x in pirads_string.split(",")]
        if max(pirads_scores) > 5:
            return int(max_pirads)
        return max(pirads_scores)
    except (IndexError, ValueError):
        return int(max_pirads)


def get_scanner_type(manufacturer_model):
    # get scanner type from manufacturer_model
    scanner = ""
    if "Vida" in manufacturer_model:
        scanner = "Vida"
    elif "Verio" in manufacturer_model:
        scanner = "Verio"
    else:
        scanner = manufacturer_model
    return scanner
