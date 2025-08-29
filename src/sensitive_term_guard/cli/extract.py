#!/usr/bin/env python3
"""
CLI tool for extracting sensitive terms from documents.
"""

import asyncio
import os

import click
from rich.console import Console
from rich.table import Table

from ..extractors import BaseSensitiveTermExtractor, LLMEnhancedSensitiveTermExtractor
from ..utils import (
    config_dict_to_objects,
    find_config_file,
    load_config_from_file,
    save_terms_to_file,
    save_terms_to_separate_files,
)

console = Console()


@click.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    "--output",
    "-o",
    default="extracted-terms.txt",
    help="Output file for extracted terms",
)
@click.option("--config", "-c", help="Configuration file path")
@click.option("--llm", is_flag=True, help="Use LLM enhancement")
@click.option("--endpoint", help="LLM endpoint URL")
@click.option("--api-key", help="LLM API key")
@click.option("--model", help="LLM model name")
@click.option("--min-score", type=float, help="Minimum sensitivity score")
@click.option("--max-terms", type=int, help="Maximum number of terms")
@click.option("--methods", multiple=True, help="Extraction methods to use")
@click.option(
    "--separate-files",
    is_flag=True,
    help="Save offline and LLM terms in separate files",
)
@click.option(
    "--include-metadata", is_flag=True, help="Include metadata in output files"
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def main(
    directory,
    output,
    config,
    llm,
    endpoint,
    api_key,
    model,
    min_score,
    max_terms,
    methods,
    separate_files,
    include_metadata,
    verbose,
):
    """Extract sensitive terms from documents in DIRECTORY."""

    if verbose:
        console.print(
            f"[green]Starting sensitive term extraction from: {directory}[/green]"
        )

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
    if min_score is not None:
        config_dict.setdefault("extraction", {})["min_score"] = min_score
    if max_terms is not None:
        config_dict.setdefault("extraction", {})["max_terms"] = max_terms
    if methods:
        config_dict.setdefault("extraction", {})["methods"] = list(methods)
    if endpoint:
        config_dict.setdefault("llm", {})["endpoint"] = endpoint
    if api_key:
        config_dict.setdefault("llm", {})["api_key"] = api_key
    if model:
        config_dict.setdefault("llm", {})["model"] = model

    # Convert config to objects
    extraction_config, llm_config, _ = config_dict_to_objects(config_dict)

    if verbose:
        console.print(f"[blue]Configuration:[/blue]")
        console.print(f"  Min score: {extraction_config.min_score}")
        console.print(f"  Max terms: {extraction_config.max_terms}")
        console.print(f"  Methods: {', '.join(extraction_config.methods)}")
        if llm:
            console.print(f"  LLM endpoint: {llm_config.endpoint}")
            console.print(f"  LLM model: {llm_config.model}")

    # Run extraction
    if llm:
        asyncio.run(
            extract_with_llm(
                directory,
                output,
                extraction_config,
                llm_config,
                separate_files,
                include_metadata,
                verbose,
            )
        )
    else:
        extract_offline(
            directory,
            output,
            extraction_config,
            separate_files,
            include_metadata,
            verbose,
        )


def extract_offline(
    directory, output, extraction_config, separate_files, include_metadata, verbose
):
    """Run offline extraction."""
    with console.status("[bold green]Extracting terms (offline mode)..."):
        extractor = BaseSensitiveTermExtractor(config=extraction_config)
        terms = extractor.extract_sensitive_terms_offline(directory)

    if terms:
        if separate_files:
            base_path = os.path.splitext(output)[0]
            files = save_terms_to_separate_files(
                [terms], None, base_path, include_metadata
            )
            console.print(f"[green]Saved terms to multiple files:[/green]")
            for file_type, filepath in files.items():
                console.print(f"  {file_type}: {filepath}")
        else:
            save_terms_to_file(terms, output, include_metadata)
            console.print(f"[green]Saved {len(terms)} terms to {output}[/green]")

        if verbose:
            display_terms_table(terms[:20], "Top 20 Extracted Terms")
    else:
        console.print("[yellow]No terms extracted[/yellow]")


async def extract_with_llm(
    directory,
    output,
    extraction_config,
    llm_config,
    separate_files,
    include_metadata,
    verbose,
):
    """Run LLM-enhanced extraction."""
    with console.status("[bold green]Extracting terms (LLM mode)..."):
        extractor = LLMEnhancedSensitiveTermExtractor(
            config=extraction_config, llm_config=llm_config
        )
        results = await extractor.extract_sensitive_terms_with_llm(directory)

    offline_terms = results["offline"]
    llm_terms = results["llm_only"]
    combined_terms = results["combined"]

    if combined_terms:
        if separate_files:
            base_path = os.path.splitext(output)[0]
            files = save_terms_to_separate_files(
                offline_terms, llm_terms, base_path, include_metadata
            )
            console.print(f"[green]Saved terms to multiple files:[/green]")
            for file_type, filepath in files.items():
                console.print(f"  {file_type}: {filepath}")
        else:
            save_terms_to_file(combined_terms, output, include_metadata)
            console.print(
                f"[green]Saved {len(combined_terms)} combined terms to {output}[/green]"
            )

        if verbose:
            console.print(f"[blue]Results summary:[/blue]")
            console.print(f"  Offline terms: {len(offline_terms)}")
            console.print(f"  LLM-only terms: {len(llm_terms)}")
            console.print(f"  Combined terms: {len(combined_terms)}")

            display_terms_table(combined_terms[:20], "Top 20 Combined Terms")
    else:
        console.print("[yellow]No terms extracted[/yellow]")


def display_terms_table(terms, title):
    """Display terms in a formatted table."""
    table = Table(title=title)
    table.add_column("Term", style="cyan")
    table.add_column("Score", justify="right", style="magenta")
    table.add_column("Frequency", justify="right", style="green")
    table.add_column("Method", style="yellow")

    for term in terms:
        table.add_row(
            term.term,
            f"{term.sensitivity_score:.1f}",
            str(term.frequency),
            term.extraction_method,
        )

    console.print(table)


if __name__ == "__main__":
    main()
