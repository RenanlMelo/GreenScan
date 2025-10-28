def format_response(success: bool, message: str, data=None):
    return {"success": success, "message": message, "data": data}
