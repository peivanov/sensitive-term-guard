#!/usr/bin/env python3
"""
CLI tool for scanning and anonymizing text using sensitive terms.
"""

import click
from rich.console import Console
from rich.table import Table

from ..scanners import DomainSensitiveScanner
from ..utils import config_dict_to_objects, find_config_file, load_config_from_file

console = Console()


@click.command()
@click.argument(
    "input_file", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option("--output", "-o", help="Output file for sanitized text")
@click.option("--terms-file", "-t", help="File containing sensitive terms")
@click.option("--config", "-c", help="Configuration file path")
@click.option(
    "--redaction-text", help="Text to use for redaction (default: [SENSITIVE])"
)
@click.option("--case-sensitive", is_flag=True, help="Case sensitive matching")
@click.option(
    "--match-type",
    type=click.Choice(["exact", "contains", "regex"]),
    help="Type of matching to use",
)
@click.option("--show-stats", is_flag=True, help="Show scanning statistics")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def main(
    input_file,
    output,
    terms_file,
    config,
    redaction_text,
    case_sensitive,
    match_type,
    show_stats,
    verbose,
):
    """Scan INPUT_FILE for sensitive terms and anonymize them."""

    if verbose:
        console.print(f"[green]Scanning file: {input_file}[/green]")

    # Load configuration
    config_dict = {}
    if config:
        config_dict = load_config_from_file(config)
    else:
        # Try to find config file automatically
        config_file = find_config_file()
        if config_file:
            config_dict = load_config_from_file(config_file)
            if verbose:
                console.print(f"[blue]Using config file: {config_file}[/blue]")

    # Override config with command line arguments
    if redaction_text:
        config_dict.setdefault("scanning", {})["redaction_text"] = redaction_text
    if case_sensitive:
        config_dict.setdefault("scanning", {})["case_sensitive"] = case_sensitive
    if match_type:
        config_dict.setdefault("scanning", {})["match_type"] = match_type

    # Convert config to objects
    _, _, scanner_config = config_dict_to_objects(config_dict)

    # Determine terms file
    if not terms_file:
        terms_file = config_dict.get("scanning", {}).get("terms_file")
        if not terms_file:
            console.print(
                "[red]Error: No terms file specified. Use --terms-file or set in config.[/red]"
            )
            return

    if verbose:
        console.print(f"[blue]Configuration:[/blue]")
        console.print(f"  Terms file: {terms_file}")
        console.print(f"  Redaction text: {scanner_config.redaction_text}")
        console.print(f"  Case sensitive: {scanner_config.case_sensitive}")
        console.print(f"  Match type: {scanner_config.match_type}")

    # Initialize scanner
    try:
        scanner = DomainSensitiveScanner(terms_file=terms_file, config=scanner_config)
    except Exception as e:
        console.print(f"[red]Error initializing scanner: {e}[/red]")
        return

    if show_stats:
        stats = scanner.get_stats()
        console.print(f"[blue]Scanner loaded with {stats['total_terms']} terms[/blue]")

    # Read input file
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        console.print(f"[red]Error reading input file: {e}[/red]")
        return

    # Scan text
    with console.status("[bold green]Scanning text..."):
        result = scanner.scan(text)

    # Display results
    if result.found_terms:
        console.print(
            f"[yellow]Found {len(result.found_terms)} sensitive terms[/yellow]"
        )
        console.print(f"[yellow]Risk score: {result.risk_score:.1f}/10[/yellow]")

        if verbose:
            display_scan_results(result)
    else:
        console.print("[green]No sensitive terms found[/green]")

    # Save output
    if output:
        try:
            with open(output, "w", encoding="utf-8") as f:
                f.write(result.sanitized_prompt)
            console.print(f"[green]Sanitized text saved to {output}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving output file: {e}[/red]")
    else:
        # Print sanitized text to stdout
        console.print("\n[bold]Sanitized text:[/bold]")
        console.print(result.sanitized_prompt)


def display_scan_results(result):
    """Display detailed scan results."""
    table = Table(title="Found Sensitive Terms")
    table.add_column("Term", style="red")
    table.add_column("Status", style="yellow")

    for term in result.found_terms:
        table.add_row(term, "REDACTED")

    console.print(table)

    console.print(f"\n[bold]Scanning details:[/bold]")
    console.print(f"  Valid: {result.is_valid}")
    console.print(f"  Risk score: {result.risk_score:.1f}")
    console.print(f"  Scan method: {result.scan_method}")


if __name__ == "__main__":
    main()
