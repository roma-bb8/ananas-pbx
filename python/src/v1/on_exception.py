def on_exception(phone_lines, uuid, peer_name, exception):
    print(f"Call with UUID {uuid} from {peer_name} caused exception: {str(exception)}\n")
    del phone_lines[uuid]
    pass
