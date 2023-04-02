import os
import sys
import typer
from pyngrok import ngrok

cli = typer.Typer()


@cli.command()
def launch(port: int = typer.Option(2023, "--port", "-p")) -> None:
    """Start a graphical UI server for the opyrator.

    The UI is auto-generated from the input- and output-schema of the given function.
    """
    # Setup a tunnel to the streamlit port 2023
    ngrok.kill()
    public_url = ngrok.connect(2023)
    print(public_url)
    # Add the current working directory to the sys path
    # This is required to resolve the opyrator path
    sys.path.append(os.getcwd())

    from toolbox.streamlit_ui import launch_ui

    launch_ui(port)
    

if __name__ == "__main__":
    cli()
