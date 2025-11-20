#!/usr/bin/env python3
"""
Development CLI for Inference Service Layer.

Provides convenient commands for common development tasks:
- setup: Install dependencies and configure environment
- test: Run tests with various options
- benchmark: Run performance benchmarks
- lint: Run code quality checks
- format: Format code
- serve: Run development server
- debug: Debug utilities
- clean: Clean temporary files

Usage:
    python scripts/dev_cli.py --help
    python scripts/dev_cli.py setup
    python scripts/dev_cli.py test --verbose
    python scripts/dev_cli.py benchmark --concurrency 10
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install: pip install typer rich")
    sys.exit(1)

app = typer.Typer(
    name="isl-dev",
    help="Inference Service Layer Development CLI",
    add_completion=False,
)

console = Console()

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent


def run_command(cmd: str, description: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with nice output."""
    console.print(f"[bold blue]→[/bold blue] {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=PROJECT_ROOT,
            check=check,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            console.print(f"[bold green]✓[/bold green] {description} complete")
        else:
            console.print(f"[bold red]✗[/bold red] {description} failed")
            if result.stderr:
                console.print(f"[red]{result.stderr}[/red]")
        return result
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]✗[/bold red] {description} failed: {e}")
        if e.stderr:
            console.print(f"[red]{e.stderr}[/red]")
        raise


@app.command()
def setup(
    dev: bool = typer.Option(False, "--dev", help="Install development dependencies"),
    redis: bool = typer.Option(True, "--redis/--no-redis", help="Start Redis server"),
):
    """
    Set up development environment.

    Installs dependencies, configures pre-commit hooks, and starts services.
    """
    console.print(Panel.fit(
        "[bold]ISL Development Setup[/bold]\n"
        "Setting up your development environment...",
        border_style="blue"
    ))

    # Check Python version
    py_version = sys.version_info
    if py_version < (3, 11):
        console.print("[red]Error: Python 3.11+ required[/red]")
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] Python {py_version.major}.{py_version.minor}.{py_version.micro}")

    # Install dependencies
    run_command("poetry install", "Installing dependencies")

    if dev:
        run_command("poetry install --with dev", "Installing dev dependencies")

    # Check if Redis is installed and running
    if redis:
        result = run_command("redis-cli ping", "Checking Redis", check=False)
        if result.returncode != 0:
            console.print("[yellow]Redis not running, attempting to start...[/yellow]")
            run_command(
                "redis-server --daemonize yes --port 6379",
                "Starting Redis",
                check=False
            )

    # Install pre-commit hooks
    pre_commit_exists = (PROJECT_ROOT / ".pre-commit-config.yaml").exists()
    if pre_commit_exists:
        run_command("poetry run pre-commit install", "Installing pre-commit hooks", check=False)

    console.print("\n[bold green]✓ Setup complete![/bold green]")
    console.print("\nNext steps:")
    console.print("  • Run tests: [cyan]python scripts/dev_cli.py test[/cyan]")
    console.print("  • Start server: [cyan]python scripts/dev_cli.py serve[/cyan]")
    console.print("  • Run benchmarks: [cyan]python scripts/dev_cli.py benchmark[/cyan]")


@app.command()
def test(
    pattern: Optional[str] = typer.Option(None, "--pattern", help="Test name pattern"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    coverage: bool = typer.Option(True, "--coverage/--no-coverage", help="Run with coverage"),
    fast: bool = typer.Option(False, "--fast", "-f", help="Skip slow tests"),
    failed: bool = typer.Option(False, "--failed", help="Only re-run failed tests"),
    marker: Optional[str] = typer.Option(None, "--marker", help="Run tests with specific marker"),
):
    """
    Run test suite with options.

    Examples:
        dev_cli.py test                         # Run all tests
        dev_cli.py test --fast                  # Skip slow tests
        dev_cli.py test --pattern preference    # Run preference tests only
        dev_cli.py test --failed                # Re-run failed tests
        dev_cli.py test --marker integration    # Run integration tests only
    """
    console.print(Panel.fit(
        "[bold]Running Tests[/bold]",
        border_style="blue"
    ))

    cmd_parts = ["poetry", "run", "pytest"]

    if verbose:
        cmd_parts.append("-v")

    if coverage:
        cmd_parts.extend(["--cov=src", "--cov-report=term", "--cov-report=html"])

    if fast:
        cmd_parts.append("-m not slow")

    if failed:
        cmd_parts.append("--lf")

    if pattern:
        cmd_parts.extend(["-k", pattern])

    if marker:
        cmd_parts.extend(["-m", marker])

    cmd = " ".join(cmd_parts)
    result = run_command(cmd, "Running tests", check=False)

    if result.returncode == 0:
        console.print("\n[bold green]✓ All tests passed![/bold green]")
    else:
        console.print("\n[bold red]✗ Some tests failed[/bold red]")
        raise typer.Exit(1)


@app.command()
def benchmark(
    concurrency: int = typer.Option(10, "--concurrency", help="Concurrent users"),
    duration: int = typer.Option(60, "--duration", help="Benchmark duration (seconds)"),
    host: str = typer.Option("http://localhost:8000", "--host", help="ISL server URL"),
    output: str = typer.Option("benchmark_results.json", "--output", help="Output file"),
):
    """
    Run performance benchmarks.

    Examples:
        dev_cli.py benchmark                       # Default: 10 users, 60s
        dev_cli.py benchmark -c 25 -d 120          # 25 users, 120s
        dev_cli.py benchmark --host http://prod    # Against production
    """
    console.print(Panel.fit(
        f"[bold]Performance Benchmark[/bold]\n"
        f"Concurrency: {concurrency} users\n"
        f"Duration: {duration}s\n"
        f"Target: {host}",
        border_style="blue"
    ))

    # Check if server is running
    import httpx
    try:
        response = httpx.get(f"{host}/health", timeout=5)
        if response.status_code != 200:
            console.print(f"[red]Error: Server at {host} not responding correctly[/red]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: Cannot connect to {host}[/red]")
        console.print(f"[red]Make sure ISL is running: python scripts/dev_cli.py serve[/red]")
        raise typer.Exit(1)

    cmd = (
        f"poetry run python benchmarks/performance_benchmark.py "
        f"--host {host} --duration {duration} --concurrency {concurrency} --output {output}"
    )

    run_command(cmd, "Running benchmark")

    console.print(f"\n[bold green]✓ Benchmark complete![/bold green]")
    console.print(f"Results saved to: [cyan]{output}[/cyan]")


@app.command()
def lint(
    fix: bool = typer.Option(False, "--fix", help="Auto-fix issues where possible"),
):
    """
    Run code quality checks (linting).

    Runs: ruff, mypy, and other linters
    """
    console.print(Panel.fit(
        "[bold]Code Quality Checks[/bold]",
        border_style="blue"
    ))

    # Run ruff
    ruff_cmd = "poetry run ruff check ."
    if fix:
        ruff_cmd += " --fix"

    result_ruff = run_command(ruff_cmd, "Running ruff", check=False)

    # Run mypy (if available)
    result_mypy = run_command(
        "poetry run mypy src --ignore-missing-imports",
        "Running mypy",
        check=False
    )

    if result_ruff.returncode == 0 and result_mypy.returncode == 0:
        console.print("\n[bold green]✓ All checks passed![/bold green]")
    else:
        console.print("\n[bold yellow]⚠ Some issues found[/bold yellow]")
        if fix:
            console.print("Run [cyan]python scripts/dev_cli.py format[/cyan] to auto-format")


@app.command()
def format(
    check: bool = typer.Option(False, "--check", help="Check formatting only, don't modify files"),
):
    """
    Format code with black and ruff.

    Examples:
        dev_cli.py format          # Format all code
        dev_cli.py format --check  # Check formatting only
    """
    console.print(Panel.fit(
        "[bold]Code Formatting[/bold]",
        border_style="blue"
    ))

    # Run black
    black_cmd = "poetry run black ."
    if check:
        black_cmd += " --check"

    result_black = run_command(black_cmd, "Running black", check=False)

    # Run ruff format
    ruff_cmd = "poetry run ruff format ."
    if check:
        ruff_cmd += " --check"

    result_ruff = run_command(ruff_cmd, "Running ruff format", check=False)

    if result_black.returncode == 0 and result_ruff.returncode == 0:
        console.print("\n[bold green]✓ Formatting complete![/bold green]")
    else:
        if check:
            console.print("\n[bold yellow]⚠ Files need formatting[/bold yellow]")
            console.print("Run [cyan]python scripts/dev_cli.py format[/cyan] to fix")
        else:
            console.print("\n[bold yellow]⚠ Some files couldn't be formatted[/bold yellow]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", help="Port to bind to"),
    reload: bool = typer.Option(True, "--reload/--no-reload", help="Enable auto-reload"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
):
    """
    Run development server.

    Examples:
        dev_cli.py serve                  # Default: http://0.0.0.0:8000
        dev_cli.py serve -p 8080          # Custom port
        dev_cli.py serve --no-reload      # Disable auto-reload
    """
    console.print(Panel.fit(
        f"[bold]Starting ISL Development Server[/bold]\n"
        f"URL: http://{host}:{port}\n"
        f"Docs: http://{host}:{port}/docs\n"
        f"Auto-reload: {'enabled' if reload else 'disabled'}",
        border_style="blue"
    ))

    # Check Redis
    result = run_command("redis-cli ping", "Checking Redis", check=False)
    if result.returncode != 0:
        console.print("[yellow]Warning: Redis not running[/yellow]")
        console.print("[yellow]Some features may not work. Start Redis:[/yellow]")
        console.print("[yellow]  redis-server --daemonize yes[/yellow]")

    cmd_parts = ["poetry", "run", "uvicorn", "src.api.main:app"]
    cmd_parts.extend(["--host", host, "--port", str(port)])

    if reload:
        cmd_parts.append("--reload")

    if debug:
        cmd_parts.append("--log-level debug")

    cmd = " ".join(cmd_parts)

    console.print("\n[bold green]Starting server...[/bold green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    try:
        subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")


@app.command()
def clean(
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Clean Python cache"),
    coverage: bool = typer.Option(True, "--coverage/--no-coverage", help="Clean coverage files"),
    build: bool = typer.Option(True, "--build/--no-build", help="Clean build artifacts"),
):
    """
    Clean temporary and generated files.

    Examples:
        dev_cli.py clean              # Clean everything
        dev_cli.py clean --no-cache   # Keep cache files
    """
    console.print(Panel.fit(
        "[bold]Cleaning Project[/bold]",
        border_style="blue"
    ))

    if cache:
        run_command(
            "find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true",
            "Cleaning Python cache",
            check=False
        )
        run_command(
            "find . -type f -name '*.pyc' -delete 2>/dev/null || true",
            "Cleaning .pyc files",
            check=False
        )

    if coverage:
        run_command(
            "rm -rf htmlcov .coverage .pytest_cache",
            "Cleaning coverage files",
            check=False
        )

    if build:
        run_command(
            "rm -rf dist build *.egg-info",
            "Cleaning build artifacts",
            check=False
        )

    # Clean Redis dump
    run_command(
        "rm -f dump.rdb",
        "Cleaning Redis dump",
        check=False
    )

    console.print("\n[bold green]✓ Cleaning complete![/bold green]")


@app.command()
def debug(
    component: str = typer.Argument(..., help="Component to debug (api, causal, preference, belief)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Debug utilities for ISL components.

    Examples:
        dev_cli.py debug api          # Debug API layer
        dev_cli.py debug preference   # Debug preference elicitation
        dev_cli.py debug causal       # Debug causal inference
    """
    console.print(Panel.fit(
        f"[bold]Debugging: {component}[/bold]",
        border_style="blue"
    ))

    debug_scripts = {
        "api": "python -m src.api.main",
        "causal": "python -m pytest tests/unit/test_causal_validator.py -v",
        "preference": "python -m pytest tests/unit/test_preference_elicitor.py -v",
        "belief": "python -m pytest tests/unit/test_belief_updater.py -v",
    }

    if component not in debug_scripts:
        console.print(f"[red]Unknown component: {component}[/red]")
        console.print(f"[yellow]Available components: {', '.join(debug_scripts.keys())}[/yellow]")
        raise typer.Exit(1)

    cmd = debug_scripts[component]
    if verbose:
        cmd += " -vv"

    run_command(cmd, f"Debugging {component}")


@app.command()
def status():
    """
    Show project status and health checks.

    Displays: test status, service health, dependencies, etc.
    """
    console.print(Panel.fit(
        "[bold]ISL Project Status[/bold]",
        border_style="blue"
    ))

    table = Table(title="Service Status", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="dim")

    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    table.add_row("Python", "[green]✓[/green]", py_version)

    # Check Poetry
    result = subprocess.run(
        "poetry --version",
        shell=True,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        poetry_version = result.stdout.strip().split()[-1] if result.stdout else "unknown"
        table.add_row("Poetry", "[green]✓[/green]", poetry_version)
    else:
        table.add_row("Poetry", "[red]✗[/red]", "Not installed")

    # Check Redis
    result = subprocess.run(
        "redis-cli ping",
        shell=True,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        table.add_row("Redis", "[green]✓[/green]", "Running")
    else:
        table.add_row("Redis", "[yellow]⚠[/yellow]", "Not running")

    # Check ISL server
    try:
        import httpx
        response = httpx.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            table.add_row("ISL Server", "[green]✓[/green]", f"v{data.get('version', 'unknown')}")
        else:
            table.add_row("ISL Server", "[yellow]⚠[/yellow]", f"HTTP {response.status_code}")
    except Exception:
        table.add_row("ISL Server", "[red]✗[/red]", "Not running")

    # Check tests
    result = subprocess.run(
        "poetry run pytest --collect-only -q 2>/dev/null | tail -1",
        shell=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT
    )
    if "test" in result.stdout:
        test_count = result.stdout.split()[0] if result.stdout else "unknown"
        table.add_row("Tests", "[green]✓[/green]", f"{test_count} collected")
    else:
        table.add_row("Tests", "[yellow]⚠[/yellow]", "Unknown")

    console.print(table)

    # Recent test results
    console.print("\n[bold]Recent Test Run:[/bold]")
    result = subprocess.run(
        "poetry run pytest --tb=no -q 2>&1 | tail -5",
        shell=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT
    )
    if result.stdout:
        console.print(result.stdout)


@app.command()
def docs(
    serve: bool = typer.Option(False, "--serve", "-s", help="Serve documentation"),
    port: int = typer.Option(8001, "--port", help="Documentation server port"),
):
    """
    Generate and optionally serve documentation.

    Examples:
        dev_cli.py docs              # Open API docs
        dev_cli.py docs --serve      # Serve documentation
    """
    if serve:
        console.print(Panel.fit(
            f"[bold]Serving Documentation[/bold]\n"
            f"API Docs: http://localhost:8000/docs\n"
            f"ReDoc: http://localhost:8000/redoc",
            border_style="blue"
        ))
        console.print("[dim]Make sure ISL server is running[/dim]")
        console.print("[cyan]python scripts/dev_cli.py serve[/cyan]\n")
    else:
        console.print("[bold]ISL Documentation Locations:[/bold]")
        console.print("\n[cyan]API Documentation:[/cyan]")
        console.print("  • Interactive: http://localhost:8000/docs (when server running)")
        console.print("  • ReDoc: http://localhost:8000/redoc")
        console.print("  • PLoT Integration: docs/PLOT_INTEGRATION_GUIDE.md")
        console.print("  • API Examples: docs/API_EXAMPLES.md")

        console.print("\n[cyan]Performance & Testing:[/cyan]")
        console.print("  • Performance Report: benchmarks/PERFORMANCE_REPORT.md")
        console.print("  • Test Analysis: docs/TEST_FAILURE_ANALYSIS.md")

        console.print("\n[cyan]Quick Links:[/cyan]")
        console.print("  • README: README.md")
        console.print("  • Contributing: CONTRIBUTING.md (if exists)")


if __name__ == "__main__":
    app()
