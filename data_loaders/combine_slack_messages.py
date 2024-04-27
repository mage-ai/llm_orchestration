def enrich_replies_and_create_document(data):
    replies = {}
    no_replies = {}

    # Create a dictionary to quickly find top level items by their ts
    ts_to_top_level_item = {item['ts']: item for item in data}
    # Create a new list for the output
    result = []

    replies_seen = {}

    # Loop through each item
    for item in data:
        item['messages'] = []
        
        if item.get('text'):
            item['messages'].append(item['text'])

        # Check if it has replies
        if 'replies' in item:
            # Loop through each reply
            for reply in item['replies']:
                # Find the matching top level item
                key = reply['ts']
                
                if key in ts_to_top_level_item:
                    matching_item = ts_to_top_level_item[key]

                    # Add 'message' to the reply
                    reply['message'] = matching_item
                    replies_seen[key] = True

                    # Add to 'messages' in the top level item
                    item['messages'].append(matching_item['text'])

        result.append(item)
        
    return [r for r in result if r['ts'] not in replies_seen]


@data_loader
def load_data(messages, *args, **kwargs):
    # document_id
    # document

    arr = [dict(
        document=message,
        document_id=message['ts'],
        text='\n'.join(message['messages']),
    ) for message in enrich_replies_and_create_document(messages)]

    return [
        arr,
    ]
