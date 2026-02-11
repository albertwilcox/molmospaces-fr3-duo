import json
import logging

import numpy as np

log = logging.getLogger(__name__)

try:
    import websockets.sync.client as ws_client
except ImportError:
    ws_client = None


class RUMClient:
    def __init__(self, host: str = "localhost", port: int = 8765, max_size: int = 50 * 1024 * 1024):
        if ws_client is None:
            raise ImportError("websockets package required. Install with: pip install websockets")
        self.uri = f"ws://{host}:{port}"
        self.websocket = ws_client.connect(self.uri, max_size=max_size)

    def _send(self, request: dict) -> dict:
        self.websocket.send(json.dumps(request))
        response = json.loads(self.websocket.recv())
        if response.get("status") == "error":
            raise RuntimeError(f"Server error: {response.get('message')}")
        return response

    def reset(self):
        self._send({"action": "reset"})

    def infer(self, observation: dict) -> np.ndarray:
        request = {
            "action": "infer",
            "observation": {
                "rgb_ego": observation["rgb_ego"].tolist(),
                "object_3d_position": observation["object_3d_position"].tolist(),
            },
        }
        return np.array(self._send(request)["result"], dtype=np.float32)

    def get_server_metadata(self) -> dict:
        response = self._send({"action": "metadata"})
        return {
            "policy_name": response.get("policy_name"),
            "checkpoint": response.get("checkpoint"),
        }

    def infer_point(
        self, rgb: np.ndarray, object_name: str = None, prompt: str = None, task: str = "pick"
    ) -> np.ndarray:
        if object_name is None and prompt is None:
            raise ValueError("Either 'object_name' or 'prompt' must be provided")
        if object_name is not None and prompt is not None:
            raise ValueError("Provide either 'object_name' or 'prompt', not both")

        request = {
            "action": "infer_point",
            "rgb": rgb.tolist(),
        }

        if object_name is not None:
            request["object_name"] = object_name
            request["task"] = task
        else:
            request["prompt"] = prompt

        response = self._send(request)
        return np.array(response["point"], dtype=np.float32)

    def close(self):
        if self.websocket:
            self.websocket.close()
            self.websocket = None
