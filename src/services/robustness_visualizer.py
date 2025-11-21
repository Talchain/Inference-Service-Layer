"""
Generate visual representations of robustness analysis.

Creates ASCII art and structured data for UI rendering of FACET results.
"""

from typing import List, Optional, Tuple

from src.models.robustness import FACETRobustnessAnalysis, InterventionRegion


class RobustnessVisualizer:
    """Generate visualizations of robustness results."""

    def generate_ascii_plot(
        self,
        analysis: FACETRobustnessAnalysis,
        variable_pair: Optional[Tuple[str, str]] = None,
    ) -> str:
        """
        Generate ASCII plot showing robust regions.

        For 2D case, shows regions in intervention space.
        For 1D case, shows intervals.

        Args:
            analysis: Robustness analysis result
            variable_pair: Optional specific variables to plot (x, y)

        Returns:
            ASCII art representation of robust regions
        """
        if not analysis.robust_regions:
            return "No robust regions found (fragile recommendation)"

        # Get first region to determine dimensionality
        first_region = analysis.robust_regions[0]
        vars_list = list(first_region.variable_ranges.keys())

        # If only 1 dimension, use 1D plot
        if len(vars_list) < 2:
            return self._generate_1d_plot(analysis)

        # Select variables to plot
        if variable_pair is None:
            variable_pair = (vars_list[0], vars_list[1] if len(vars_list) > 1 else vars_list[0])

        # Check if this is actually 2D
        if len(vars_list) == 1 or variable_pair[0] == variable_pair[1]:
            return self._generate_1d_plot(analysis)

        return self._generate_2d_plot(analysis, variable_pair)

    def _generate_1d_plot(self, analysis: FACETRobustnessAnalysis) -> str:
        """
        Generate 1D interval plot.

        Args:
            analysis: Robustness analysis result

        Returns:
            ASCII representation of 1D intervals
        """
        lines = []
        lines.append("\nRobust Intervals:")
        lines.append("=" * 60)

        for i, region in enumerate(analysis.robust_regions):
            var = list(region.variable_ranges.keys())[0]
            min_val, max_val = region.variable_ranges[var]

            # Visual representation
            bar_length = 40
            range_width = max_val - min_val
            bar = "#" * max(1, int(bar_length * (range_width / (max_val + min_val))))

            lines.append(f"\nRegion {i+1}: {var}")
            lines.append(f"  Range: [{min_val:.2f}, {max_val:.2f}]")
            lines.append(f"  Visual: {bar}")
            lines.append(f"  Width: {range_width:.2f}")

        lines.append(f"\nOverall Robustness Score: {analysis.robustness_score:.2f}")
        lines.append(f"Total Regions Found: {analysis.region_count}")

        return "\n".join(lines)

    def _generate_2d_plot(
        self,
        analysis: FACETRobustnessAnalysis,
        variable_pair: Tuple[str, str],
    ) -> str:
        """
        Generate 2D ASCII plot.

        Args:
            analysis: Robustness analysis result
            variable_pair: Variables to plot (x_var, y_var)

        Returns:
            ASCII 2D plot
        """
        var_x, var_y = variable_pair

        lines = []
        lines.append(f"\nRobust Regions: {var_x} vs {var_y}")
        lines.append("=" * 60)

        # Get bounds across all regions
        all_x_vals = []
        all_y_vals = []

        for region in analysis.robust_regions:
            if var_x in region.variable_ranges and var_y in region.variable_ranges:
                x_min, x_max = region.variable_ranges[var_x]
                y_min, y_max = region.variable_ranges[var_y]
                all_x_vals.extend([x_min, x_max])
                all_y_vals.extend([y_min, y_max])

        if not all_x_vals:
            return "Cannot visualize: selected variables not in regions"

        # Create grid
        grid_width = 50
        grid_height = 20

        x_min_global = min(all_x_vals)
        x_max_global = max(all_x_vals)
        y_min_global = min(all_y_vals)
        y_max_global = max(all_y_vals)

        # Initialize grid
        grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]

        # Fill robust regions with '#'
        for region in analysis.robust_regions:
            if var_x in region.variable_ranges and var_y in region.variable_ranges:
                x_min, x_max = region.variable_ranges[var_x]
                y_min, y_max = region.variable_ranges[var_y]

                # Map to grid coordinates
                x_start = int(
                    (x_min - x_min_global) / (x_max_global - x_min_global) * (grid_width - 1)
                )
                x_end = int(
                    (x_max - x_min_global) / (x_max_global - x_min_global) * (grid_width - 1)
                )
                y_start = int(
                    (y_min - y_min_global) / (y_max_global - y_min_global) * (grid_height - 1)
                )
                y_end = int(
                    (y_max - y_min_global) / (y_max_global - y_min_global) * (grid_height - 1)
                )

                # Fill region
                for y in range(y_start, y_end + 1):
                    for x in range(x_start, x_end + 1):
                        if 0 <= y < grid_height and 0 <= x < grid_width:
                            grid[y][x] = '#'

        # Add y-axis label
        lines.append(f"\n{var_y}  ^")

        # Render grid (flip y-axis for natural orientation)
        for y in range(grid_height - 1, -1, -1):
            # Add y-axis value occasionally
            if y % 5 == 0:
                y_val = y_min_global + (y / grid_height) * (y_max_global - y_min_global)
                label = f"{y_val:6.1f} |"
            else:
                label = "       |"
            lines.append(label + ''.join(grid[y]))

        # Add x-axis
        lines.append("       +" + "-" * grid_width + f"> {var_x}")
        lines.append(f"       {x_min_global:.1f}" + " " * (grid_width - 20) + f"{x_max_global:.1f}")

        lines.append(f"\nLegend:")
        lines.append(f"  # = Robust region achieving target")
        lines.append(f"\nRobustness score: {analysis.robustness_score:.2f}")
        lines.append(f"Regions found: {analysis.region_count}")

        return "\n".join(lines)

    def generate_summary_table(self, analysis: FACETRobustnessAnalysis) -> str:
        """
        Generate summary table for logging/display.

        Args:
            analysis: Robustness analysis result

        Returns:
            Formatted summary table
        """
        lines = []
        lines.append("\n╔═══════════════════════════════════════════════════╗")
        lines.append("║       FACET ROBUSTNESS ANALYSIS SUMMARY           ║")
        lines.append("╠═══════════════════════════════════════════════════╣")

        # Status
        status_display = analysis.status.upper()
        status_line = f"║ Status:           {status_display:<32} ║"
        lines.append(status_line)

        # Robustness Score
        score_display = f"{analysis.robustness_score:.3f}"
        score_line = f"║ Robustness Score: {score_display:<32} ║"
        lines.append(score_line)

        # Regions Found
        regions_display = str(analysis.region_count)
        regions_line = f"║ Regions Found:    {regions_display:<32} ║"
        lines.append(regions_line)

        # Fragile indicator
        fragile_display = "YES ⚠️ " if analysis.is_fragile else "NO ✓"
        fragile_line = f"║ Fragile:          {fragile_display:<32} ║"
        lines.append(fragile_line)

        # Samples
        samples_display = f"{analysis.samples_successful}/{analysis.samples_tested}"
        samples_line = f"║ Samples (success/total): {samples_display:<23} ║"
        lines.append(samples_line)

        lines.append("╚═══════════════════════════════════════════════════╝")

        # Fragility warnings
        if analysis.fragility_reasons:
            lines.append("\n⚠️  Fragility Warnings:")
            for reason in analysis.fragility_reasons:
                # Wrap long reasons
                if len(reason) > 55:
                    words = reason.split()
                    current_line = "  • "
                    for word in words:
                        if len(current_line) + len(word) + 1 > 55:
                            lines.append(current_line)
                            current_line = "    " + word + " "
                        else:
                            current_line += word + " "
                    if current_line.strip():
                        lines.append(current_line)
                else:
                    lines.append(f"  • {reason}")

        # Outcome guarantees
        if analysis.outcome_guarantees:
            lines.append("\n✓ Outcome Guarantees:")
            for outcome_var, guarantee in analysis.outcome_guarantees.items():
                lines.append(
                    f"  • {outcome_var}: "
                    f"[{guarantee.minimum:.1f}, {guarantee.maximum:.1f}] "
                    f"({guarantee.confidence*100:.0f}% confidence)"
                )

        # Robust regions detail
        if analysis.robust_regions and len(analysis.robust_regions) <= 3:
            lines.append("\n📍 Robust Regions:")
            for i, region in enumerate(analysis.robust_regions):
                lines.append(f"\n  Region {i+1}:")
                for var, (min_val, max_val) in region.variable_ranges.items():
                    lines.append(f"    {var}: [{min_val:.2f}, {max_val:.2f}]")

        # Interpretation
        lines.append(f"\n💡 Interpretation:")
        # Wrap interpretation
        interpretation_lines = self._wrap_text(analysis.interpretation, 58)
        for line in interpretation_lines:
            lines.append(f"   {line}")

        # Recommendation
        lines.append(f"\n🎯 Recommendation:")
        recommendation_lines = self._wrap_text(analysis.recommendation, 58)
        for line in recommendation_lines:
            lines.append(f"   {line}")

        return "\n".join(lines)

    def _wrap_text(self, text: str, width: int) -> List[str]:
        """
        Wrap text to specified width.

        Args:
            text: Text to wrap
            width: Maximum line width

        Returns:
            List of wrapped lines
        """
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= width:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines
