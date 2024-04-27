from typing import Any, List, Optional

import pandas as pd


def bucket_items(
    items: List[Any],
    num_buckets: Optional[int] = None,
    max_num_buckets: Optional[int] = None,
    max_items_per_bucket: Optional[int] = None,
    max_bytes_per_bucket: Optional[int] = None,
    override_on_max: bool = False,
) -> List[List[Any]]:
    """
    Distributes items into buckets based on the specified criteria. Prioritizes conditions
    in the order of num_buckets, conditions (max_items_per_bucket, max_bytes_per_bucket),
    and then max_num_buckets. Allows overriding other parameters if all maximums are exceeded
    when override_on_max is True.
    """
    def item_fits_in_bucket(bucket: List[Any], item: Any, override=False) -> bool:
        if not override:
            if max_items_per_bucket is not None and len(bucket) >= max_items_per_bucket:
                return False
            if max_bytes_per_bucket is not None:
                current_bucket_byte_size = sum(len(str(x).encode('utf-8')) for x in bucket)
                item_byte_size = len(str(item).encode('utf-8'))
                if current_bucket_byte_size + item_byte_size > max_bytes_per_bucket:
                    return False
        return True

    buckets = [[] for _ in range(num_buckets)] if num_buckets else []
    item_count_in_last_bucket = 0

    for item in items:
        fit_found = False
        for bucket in buckets:
            if item_fits_in_bucket(bucket, item):
                if not item_count_in_last_bucket:
                    print(f"Creating new bucket and adding items. Current total buckets: {len(buckets)}.")
                bucket.append(item)
                item_count_in_last_bucket += 1
                fit_found = True
                break

        if not fit_found:
            if item_count_in_last_bucket:
                print(f"{item_count_in_last_bucket} items added to the last bucket.")
                item_count_in_last_bucket = 0
            if not num_buckets and (max_num_buckets is None or len(buckets) < max_num_buckets) or override_on_max:
                buckets.append([item])
                item_count_in_last_bucket += 1
            elif max_num_buckets is not None and len(buckets) >= max_num_buckets and not override_on_max:
                print(f"Warning: Item '{item}' couldn't be placed due to exceeding the max number of buckets.")
                break

    if item_count_in_last_bucket:
        print(f"{item_count_in_last_bucket} items added to the last bucket.")

    print(f"Total buckets created: {len(buckets)}.")
    return buckets


def bucket_dataframes(
    dataframes: List[pd.DataFrame],
    num_buckets: Optional[int] = None,
    max_num_buckets: Optional[int] = None,
    max_items_per_bucket: Optional[int] = None,
    # 0.5 GBs in memory
    # 2 GBs on disk
    max_bytes_per_bucket: Optional[int] = (1.9 * 1024**3) / 8,
    override_on_max: bool = False,
) -> List[List[pd.DataFrame]]:
    """
    Distributes DataFrames into buckets based on the specified criteria. Prioritizes conditions
    in the order of num_buckets, conditions (max_items_per_bucket, max_bytes_per_bucket),
    and then max_num_buckets. Allows overriding other parameters if all maximums are exceeded
    when override_on_max is True.
    """
    def dataframe_fits_in_bucket(bucket: List[pd.DataFrame], dataframe: pd.DataFrame, override=False) -> bool:
        if not override:
            if max_items_per_bucket is not None and len(bucket) >= max_items_per_bucket:
                return False
            if max_bytes_per_bucket is not None:
                current_bucket_byte_size = sum(df.memory_usage(deep=True).sum() for df in bucket)
                dataframe_byte_size = dataframe.memory_usage(deep=True).sum()
                if current_bucket_byte_size + dataframe_byte_size > max_bytes_per_bucket:
                    return False
        return True

    buckets = [[] for _ in range(num_buckets)] if num_buckets else []

    for dataframe in dataframes:
        fit_found = False
        for bucket in buckets:
            if dataframe_fits_in_bucket(bucket, dataframe):
                bucket.append(dataframe)
                fit_found = True
                break

        if not fit_found:
            if not num_buckets and (max_num_buckets is None or len(buckets) < max_num_buckets) or override_on_max:
                buckets.append([dataframe])
            elif max_num_buckets is not None and len(buckets) >= max_num_buckets and not override_on_max:
                print(f"Warning: A DataFrame couldn't be placed due to exceeding the max number of buckets.")
                break  # or continue, depending on whether you want to stop adding DataFrames or skip the unplaceable ones

    return buckets
