# MeetingGenie - Shell Layout
# Top-level layout: header + NavigationRail + content area

import flet as ft
from app_state import AppState


# Brand colours
BRAND_BLUE = "#0A84FF"
BRAND_BLUE_DARK = "#0066CC"


def build_shell(page: ft.Page, state: AppState, pages: dict[str, ft.Control]):
    """Build the shell layout and attach it to the page.

    pages: mapping of page keys ("transcribe", "queue", "editor", "settings")
           to Control instances returned by each page builder.
    """

    # --- Dark / Light toggle -------------------------------------------
    def toggle_theme(e):
        if page.theme_mode == ft.ThemeMode.DARK:
            page.theme_mode = ft.ThemeMode.LIGHT
            theme_btn.icon = ft.Icons.DARK_MODE
            theme_btn.tooltip = "Switch to dark mode"
        else:
            page.theme_mode = ft.ThemeMode.DARK
            theme_btn.icon = ft.Icons.LIGHT_MODE
            theme_btn.tooltip = "Switch to light mode"
        page.update()

    theme_btn = ft.IconButton(
        icon=ft.Icons.LIGHT_MODE,
        tooltip="Switch to light mode",
        on_click=toggle_theme,
    )

    # --- Header --------------------------------------------------------
    header = ft.Container(
        content=ft.Row(
            [
                ft.Row(
                    [
                        ft.Icon(ft.Icons.MIC, color=BRAND_BLUE, size=28),
                        ft.Text(
                            "MeetingGenie",
                            size=24,
                            weight=ft.FontWeight.BOLD,
                            color=BRAND_BLUE,
                        ),
                        ft.Container(
                            content=ft.Row([
                                ft.Icon(
                                    ft.Icons.MEMORY if state.compute_device in ("metal", "cuda") else ft.Icons.COMPUTER,
                                    size=14,
                                    color="#4CAF50" if state.compute_device in ("metal", "cuda") else ft.Colors.ON_SURFACE_VARIANT,
                                ),
                                ft.Text(
                                    state.get_device_label(),
                                    size=11,
                                    color="#4CAF50" if state.compute_device in ("metal", "cuda") else ft.Colors.ON_SURFACE_VARIANT,
                                ),
                            ], spacing=4, tight=True),
                            border=ft.border.all(1, "#4CAF50" if state.compute_device in ("metal", "cuda") else ft.Colors.OUTLINE_VARIANT),
                            border_radius=12,
                            padding=ft.padding.symmetric(horizontal=8, vertical=3),
                        ),
                    ],
                    spacing=8,
                ),
                ft.Row([
                    state.input_device_dropdown if hasattr(state, 'input_device_dropdown') else ft.Container(),
                    state.live_btn if hasattr(state, 'live_btn') else ft.Container(),
                    theme_btn,
                ], spacing=8),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        ),
        padding=ft.padding.symmetric(horizontal=20, vertical=10),
    )

    # --- Content area (switches between pages) -------------------------
    page_keys = list(pages.keys())
    content_stack = ft.Column(
        [pages[page_keys[0]]],
        expand=True,
    )

    def on_nav_change(e):
        idx = e.control.selected_index
        key = page_keys[idx]
        content_stack.controls = [pages[key]]
        content_stack.update()

    # --- NavigationRail ------------------------------------------------
    rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=80,
        min_extended_width=180,
        group_alignment=-0.9,
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.Icons.MIC_OUTLINED,
                selected_icon=ft.Icons.MIC,
                label="Transcribe",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.QUEUE_OUTLINED,
                selected_icon=ft.Icons.QUEUE,
                label="Queue",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.EDIT_NOTE_OUTLINED,
                selected_icon=ft.Icons.EDIT_NOTE,
                label="Editor",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.SETTINGS_OUTLINED,
                selected_icon=ft.Icons.SETTINGS,
                label="Settings",
            ),
        ],
        on_change=on_nav_change,
    )

    # --- Compose layout ------------------------------------------------
    body = ft.Row(
        [
            rail,
            ft.VerticalDivider(width=1),
            content_stack,
        ],
        expand=True,
    )

    page.add(
        ft.Column(
            [header, ft.Divider(height=1), body],
            expand=True,
            spacing=0,
        )
    )
