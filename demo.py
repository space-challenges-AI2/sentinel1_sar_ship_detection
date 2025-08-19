#!/usr/bin/env python3
"""
SAR Ship Detection Pipeline - Demo
Author: @amanarora9848
"""

import time
import os
import threading
from pathlib import Path
from utils.pipeline import PipelineCoordinator

# Install rich: pip install rich
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.columns import Columns

def create_status_display():
    """Create a rich status display layout"""
    layout = Layout()
    
    # Header
    header = Panel(
        "[bold blue]SAR Ship Detection Pipeline[/bold blue]\n"
        "[cyan]AI-powered ship detection in satellite imagery[/cyan]",
        style="bold white on blue"
    )
    
    # Main content area
    layout.split_column(
        Layout(header, size=5),
        Layout(name="main", ratio=1)
    )
    
    return layout

def create_pipeline_status(coordinator):
    """Create a rich table showing pipeline status"""
    table = Table(title="Pipeline Status", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    
    status = coordinator.get_status()
    stats = status['statistics']
    
    table.add_row("Pipeline", status['status'], f"Running: {time.strftime('%H:%M:%S')}")
    table.add_row("Ingest Directory", "Active", status['ingest_directory'])
    table.add_row("AI Model", "Loaded", status['weights_path'])
    table.add_row("Denoising", "Active", status['denoise_method'])
    table.add_row("Images Processed", str(stats['images_processed']), f"Total: {stats['total_detections']} ships")
    table.add_row("Processing Time", f"{stats['processing_time_total']:.1f}s", "Real-time")
    
    return table

def run_demo():
    console = Console()
    
    # Clear screen and show header
    console.clear()
    console.print(Panel.fit(
        "[bold blue] SAR Ship Detection Pipeline - Demo[/bold blue]",
        style="bold white on blue"
    ))
    
    # Add denoising control
    console.print("\n[bold cyan]Denoising Options:[/bold cyan]")
    console.print("1. None (no denoising)")
    console.print("2. FABF (default)")
    
    choice = console.input("\n[bold yellow]Select denoising method (1-2, default 2):[/bold yellow] ").strip()
    
    # Set denoising based on choice
    if choice == "1":
        denoise_method = "none"
        denoise_probability = 0.0
    else:  # default
        denoise_method = "fabf"
        denoise_probability = 1.0
    
    console.print(f"\n[green] Using denoising: {denoise_method}[/green]")
    
    # Initialize pipeline
    with console.status("[bold green]Initializing pipeline...", spinner="dots"):
        coordinator = PipelineCoordinator()
        coordinator.denoise_method = denoise_method
        coordinator.denoise_probability = denoise_probability
        # Override the weights path to use experiment4
        coordinator.weights_path = Path("runs/train/experiment4/weights/best.pt")
    
    console.print("[green] Pipeline initialized successfully![/green]")
    
    # Start pipeline
    with console.status("[bold green]Starting pipeline...", spinner="dots"):
        coordinator.start()
    
    # Start continuous processing in background
    pipeline_thread = threading.Thread(
        target=coordinator.run_continuous, 
        args=(2.0,),
        daemon=True
    )
    pipeline_thread.start()
    
    console.print("[green] Pipeline started successfully![/green]")
    
    # Copy source images and count them
    source_dir = Path("source")
    ingest_dir = Path(coordinator.get_status()['ingest_directory'])
    total_images = 0
    
    if source_dir.exists() and source_dir.is_dir():
        image_files = [f for f in source_dir.iterdir() if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}]
        if image_files:
            total_images = len(image_files)
            with console.status(f"[bold green]Copying {total_images} images to ingest...", spinner="dots"):
                import shutil
                for img_path in image_files:
                    shutil.copy2(img_path, ingest_dir)
            console.print(f"[green] Copied {total_images} images to ingest directory[/green]")
        else:
            console.print(f"[yellow] No images found in {source_dir}[/yellow]")
            total_images = 0
    else:
        console.print(f"[yellow] Source directory {source_dir} not found[/yellow]")
        total_images = 0
    
    # Create live display
    layout = create_status_display()
    
    # Wait for pipeline to process all images
    if total_images > 0:
        console.print(f"\n[bold yellow] Waiting for pipeline to process all {total_images} images...[/bold yellow]")
    
    with Live(layout, refresh_per_second=2, screen=True) as live:
        # Wait for all images to be processed
        max_wait_time = 120  # Maximum wait time in seconds (2 minutes)
        start_wait = time.time()
        
        while time.time() - start_wait < max_wait_time:
            # Update status
            status_table = create_pipeline_status(coordinator)
            layout["main"].update(status_table)
            
            # Check if all images have been processed
            status = coordinator.get_status()
            stats = status['statistics']
            
            # Check if all images have been processed
            if stats['images_processed'] >= total_images:
                console.print(f"\n[green] All {total_images} images processed successfully![/green]")
                break
            
            # Show progress
            progress = min(stats['images_processed'] / total_images * 100, 100) if total_images > 0 else 0
            console.print(f"\r[cyan]Progress: {stats['images_processed']}/{total_images} images processed ({progress:.1f}%)[/cyan]", end="")
            
            time.sleep(2)
        else:
            console.print(f"\n[yellow] Timeout waiting for all images to be processed. Processed: {stats['images_processed']}/{total_images}[/yellow]")
    
    # Final results
    final_status = coordinator.get_status()
    final_stats = final_status['statistics']
    
    console.print("\n[bold green] Final Results:[/bold green]")
    console.print(Panel(
        f"[bold]Total Images:[/bold] {final_stats['images_processed']}/{total_images}\n"
        f"[bold]Total Ships:[/bold] {final_stats['total_detections']}\n"
        f"[bold]Processing Time:[/bold] {final_stats['processing_time_total']:.1f}s",
        title="Pipeline Summary",
        style="bold green"
    ))
    
    console.print("\n[bold green] Demo complete![/bold green]")
    coordinator.stop()

if __name__ == "__main__":
    run_demo()