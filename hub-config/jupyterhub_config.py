c = get_config()  # noqa: F821

c.JupyterHub.bind_url = "http://0.0.0.0:8000"
c.Authenticator.admin_users = {"rift"}
c.JupyterHub.allow_named_servers = True
c.Spawner.default_url = "/lab"
c.Spawner.notebook_dir = "/workspace"
c.JupyterHub.log_level = "INFO"
