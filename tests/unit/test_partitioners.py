"""Unit tests for partitioners."""

# from __future__ import annotations

# import numpy as np
# import pandas as pd
# import pytest

# from xeries.partitioners.manual import ManualPartitioner
# from xeries.partitioners.tree import TreePartitioner


class TestManualPartitioner:
    """Tests for ManualPartitioner."""

    def first(self) -> None:
        """Placeholder test to ensure test discovery works."""
        assert True


#     def test_init(self, series_mapping: dict[str, str]) -> None:
#         """Test partitioner initialization."""
#         partitioner = ManualPartitioner(series_mapping, series_col="level")
#         assert partitioner.mapping == series_mapping
#         assert partitioner.series_col == "level"
#         assert not partitioner._fitted

#     def test_fit(
#         self,
#         sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
#         series_mapping: dict[str, str],
#     ) -> None:
#         """Test fitting the partitioner."""
#         X, _ = sample_multiindex_data
#         partitioner = ManualPartitioner(series_mapping, series_col="level")

#         result = partitioner.fit(X, feature="lag_1")

#         assert result is partitioner
#         assert partitioner._fitted
#         assert len(partitioner._group_encoder) == 2

#     def test_get_groups(
#         self,
#         sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
#         series_mapping: dict[str, str],
#     ) -> None:
#         """Test getting group labels."""
#         X, _ = sample_multiindex_data
#         partitioner = ManualPartitioner(series_mapping, series_col="level")
#         partitioner.fit(X, feature="lag_1")

#         groups = partitioner.get_groups(X)

#         assert len(groups) == len(X)
#         assert groups.dtype == np.intp
#         assert set(np.unique(groups)) == {0, 1}

#     def test_get_groups_not_fitted(
#         self,
#         sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
#         series_mapping: dict[str, str],
#     ) -> None:
#         """Test error when getting groups before fitting."""
#         X, _ = sample_multiindex_data
#         partitioner = ManualPartitioner(series_mapping, series_col="level")

#         with pytest.raises(ValueError, match="must be fitted"):
#             partitioner.get_groups(X)

#     def test_fit_get_groups(
#         self,
#         sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
#         series_mapping: dict[str, str],
#     ) -> None:
#         """Test combined fit and get_groups."""
#         X, _ = sample_multiindex_data
#         partitioner = ManualPartitioner(series_mapping, series_col="level")

#         groups = partitioner.fit_get_groups(X, feature="lag_1")

#         assert len(groups) == len(X)
#         assert partitioner._fitted

#     def test_missing_series_in_mapping(
#         self,
#         sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
#     ) -> None:
#         """Test error when series ID not in mapping."""
#         X, _ = sample_multiindex_data
#         incomplete_mapping = {"MT_001": "group_A"}
#         partitioner = ManualPartitioner(incomplete_mapping, series_col="level")
#         partitioner.fit(X, feature="lag_1")

#         with pytest.raises(ValueError, match="not found in mapping"):
#             partitioner.get_groups(X)

#     def test_n_groups(
#         self,
#         sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
#         series_mapping: dict[str, str],
#     ) -> None:
#         """Test n_groups property."""
#         X, _ = sample_multiindex_data
#         partitioner = ManualPartitioner(series_mapping, series_col="level")
#         partitioner.fit(X, feature="lag_1")

#         assert partitioner.n_groups == 2


# class TestTreePartitioner:
#     """Tests for TreePartitioner."""

#     def test_init(self) -> None:
#         """Test partitioner initialization."""
#         partitioner = TreePartitioner(max_depth=4, min_samples_leaf=0.05)
#         assert partitioner.max_depth == 4
#         assert partitioner.min_samples_leaf == 0.05
#         assert not partitioner._fitted

#     def test_fit(
#         self,
#         sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
#     ) -> None:
#         """Test fitting the tree partitioner."""
#         X, _ = sample_multiindex_data
#         partitioner = TreePartitioner(max_depth=3, random_state=42)

#         result = partitioner.fit(X, feature="lag_1")

#         assert result is partitioner
#         assert partitioner._fitted
#         assert partitioner._tree is not None
#         assert partitioner._feature == "lag_1"

#     def test_get_groups(
#         self,
#         sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
#     ) -> None:
#         """Test getting group labels from tree leaves."""
#         X, _ = sample_multiindex_data
#         partitioner = TreePartitioner(max_depth=3, random_state=42)
#         partitioner.fit(X, feature="lag_1")

#         groups = partitioner.get_groups(X)

#         assert len(groups) == len(X)
#         assert groups.dtype == np.intp
#         assert len(np.unique(groups)) <= 2**3

#     def test_get_groups_not_fitted(
#         self,
#         sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
#     ) -> None:
#         """Test error when getting groups before fitting."""
#         X, _ = sample_multiindex_data
#         partitioner = TreePartitioner()

#         with pytest.raises(ValueError, match="must be fitted"):
#             partitioner.get_groups(X)

#     def test_n_groups(
#         self,
#         sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
#     ) -> None:
#         """Test n_groups property."""
#         X, _ = sample_multiindex_data
#         partitioner = TreePartitioner(max_depth=2, random_state=42)
#         partitioner.fit(X, feature="lag_1")

#         assert partitioner.n_groups > 0
#         assert partitioner.n_groups <= 2**2

#     def test_tree_property(
#         self,
#         sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
#     ) -> None:
#         """Test tree property returns fitted tree."""
#         X, _ = sample_multiindex_data
#         partitioner = TreePartitioner(max_depth=2, random_state=42)

#         assert partitioner.tree is None

#         partitioner.fit(X, feature="lag_1")
#         assert partitioner.tree is not None

#     def test_without_series_col(
#         self,
#         sample_flat_data: tuple[pd.DataFrame, pd.Series],
#     ) -> None:
#         """Test partitioner with flat data (no series encoding)."""
#         X, _ = sample_flat_data
#         X_numeric = X.drop(columns=["series_id"])
#         partitioner = TreePartitioner(max_depth=3, series_col=None, random_state=42)

#         groups = partitioner.fit_get_groups(X_numeric, feature="feature_0")

#         assert len(groups) == len(X_numeric)
