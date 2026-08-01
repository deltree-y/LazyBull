import pandas as pd
import pytest

from src.lazybull.ml.train_core import add_blended_return_label


def test_zero_weight_keeps_neutral_label_without_new_column():
    df = pd.DataFrame({"neu_y_ret_20": [0.1], "y_ret_20": [0.3]})

    label_column = add_blended_return_label(df, "neu_y_ret_20", 0.0)

    assert label_column == "neu_y_ret_20"
    assert "y_blend_ret_20" not in df.columns


def test_positive_weight_adds_blended_label():
    df = pd.DataFrame(
        {
            "neu_y_ret_20": [0.1, -0.2],
            "y_ret_20": [0.3, 0.2],
        }
    )

    label_column = add_blended_return_label(df, "neu_y_ret_20", 0.25)

    assert label_column == "y_blend_ret_20"
    assert df[label_column].tolist() == pytest.approx([0.15, -0.1])


@pytest.mark.parametrize("blend_weight", [-0.01, 1.01])
def test_invalid_weight_is_rejected(blend_weight):
    df = pd.DataFrame({"neu_y_ret_20": [0.1], "y_ret_20": [0.3]})

    with pytest.raises(ValueError, match="必须在"):
        add_blended_return_label(df, "neu_y_ret_20", blend_weight)


def test_positive_weight_requires_neutral_and_raw_labels():
    df = pd.DataFrame({"neu_y_ret_20": [0.1]})

    with pytest.raises(ValueError, match="y_ret_20"):
        add_blended_return_label(df, "neu_y_ret_20", 0.25)
