from pathlib import Path

import yaml


def test_render_blueprint_wires_api_frontend_and_database():
    blueprint = yaml.safe_load(Path("render.yaml").read_text(encoding="utf-8"))
    services = {service["name"]: service for service in blueprint["services"]}
    api = services["nyanja-credit-risk-api"]
    frontend = services["nyanja-credit-risk-ui"]
    database = blueprint["databases"][0]

    assert api["runtime"] == "docker"
    assert api["healthCheckPath"] == "/health"
    assert api["region"] == database["region"] == "frankfurt"
    assert api["envVars"][0]["fromDatabase"]["name"] == database["name"]
    assert frontend["envVars"][0]["value"] == (
        "https://nyanja-credit-risk-api.onrender.com"
    )
    assert api["envVars"][1]["value"] == (
        "https://nyanja-credit-risk-ui.onrender.com"
    )
    assert frontend["staticPublishPath"] == "./frontend/dist"
    assert database["plan"] == "basic-256mb"
