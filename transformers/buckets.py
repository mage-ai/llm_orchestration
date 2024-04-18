def evenly_distribute_to_buckets(input_list, num_buckets):
    # The total length of the input list
    total_items = len(input_list)
    
    # Determine the base size of each bucket
    base_bucket_size = total_items // num_buckets
    
    # Determine how many buckets should have an extra item
    # to account for when total_items isn't evenly divisible by num_buckets
    buckets_with_extra = total_items % num_buckets
    
    buckets = []
    start_index = 0
    
    for i in range(num_buckets):
        # Determine if the current bucket is to have an extra item
        extra_item = 1 if i < buckets_with_extra else 0
        # Calculate the end index for slicing the input_list
        end_index = start_index + base_bucket_size + extra_item
        # Append the slice of input_list to buckets
        buckets.append(input_list[start_index:end_index])
        # Update start_index for the next bucket
        start_index = end_index
    
    return buckets


@transformer
def transform(data, *args, **kwargs):
    return [
        evenly_distribute_to_buckets(data, 100),
    ]