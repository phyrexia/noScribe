# MeetingGenie - Shell Layout
# Top-level layout: header + NavigationRail + content area

import flet as ft
from app_state import AppState
from event_bus import EventType
from config import APP_VERSION


# Brand colours
BRAND_BLUE = "#0A84FF"
BRAND_BLUE_DARK = "#0066CC"


def _is_accelerated(backend: str) -> bool:
    return backend in ("mlx-metal", "cuda")


def _label_is_accelerated(label: str) -> bool:
    """Heuristic for role-specific labels (e.g. pyannote 'Metal/GPU (MPS)')."""
    if not label:
        return False
    low = label.lower()
    return any(k in low for k in ("metal", "mps", "cuda", "gpu", "mlx"))


def _build_badge_content(state: AppState) -> ft.Row:
    accel = _is_accelerated(state.compute_backend)
    color = "#4CAF50" if accel else ft.Colors.ON_SURFACE_VARIANT
    icon = ft.Icons.MEMORY if accel else ft.Icons.COMPUTER
    return ft.Row([
        ft.Icon(icon, size=14, color=color),
        ft.Text(state.compute_device_label, size=11, color=color),
    ], spacing=4, tight=True)


def _badge_border_color(state: AppState):
    return "#4CAF50" if _is_accelerated(state.compute_backend) else ft.Colors.OUTLINE_VARIANT


def _build_role_chip(prefix: str, label: str) -> ft.Container:
    """Small chip showing 'Prefix: Label' with accent if the label looks accelerated."""
    accel = _label_is_accelerated(label)
    color = "#4CAF50" if accel else ft.Colors.ON_SURFACE_VARIANT
    border = "#4CAF50" if accel else ft.Colors.OUTLINE_VARIANT
    icon = ft.Icons.MEMORY if accel else ft.Icons.COMPUTER
    return ft.Container(
        content=ft.Row([
            ft.Icon(icon, size=12, color=color),
            ft.Text(f"{prefix}: {label}", size=10, color=color),
        ], spacing=4, tight=True),
        border=ft.border.all(1, border),
        border_radius=10,
        padding=ft.padding.symmetric(horizontal=6, vertical=2),
    )


def _build_role_chips_row(state: AppState) -> ft.Row:
    """Render a row of per-role chips, hiding roles that haven't reported yet."""
    chips = []
    diar = state.compute_devices.get('diarization', '')
    trans = state.compute_devices.get('transcription', '')
    if diar:
        chips.append(_build_role_chip("Diar", diar))
    if trans:
        chips.append(_build_role_chip("Trans", trans))
    return ft.Row(chips, spacing=4, tight=True)


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

    # --- Compute device badge ----------------------------------------
    device_badge = ft.Container(
        content=_build_badge_content(state),
        border=ft.border.all(1, _badge_border_color(state)),
        border_radius=12,
        padding=ft.padding.symmetric(horizontal=8, vertical=3),
    )

    # Per-role chips (diarization / transcription) — hidden until reported.
    role_chips_container = ft.Container(content=_build_role_chips_row(state))

    def _on_device_update(_payload):
        device_badge.content = _build_badge_content(state)
        device_badge.border = ft.border.all(1, _badge_border_color(state))
        role_chips_container.content = _build_role_chips_row(state)
        try:
            device_badge.update()
        except Exception:
            pass
        try:
            role_chips_container.update()
        except Exception:
            pass

    state.bus.subscribe(EventType.DEVICE_UPDATE, _on_device_update)

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
                            content=ft.Text(
                                f"v{APP_VERSION}",
                                size=11,
                                color=ft.Colors.ON_SURFACE_VARIANT,
                                weight=ft.FontWeight.W_500,
                            ),
                            padding=ft.padding.symmetric(horizontal=8, vertical=2),
                            bgcolor=ft.Colors.SURFACE_VARIANT,
                            border_radius=10,
                        ),
                        device_badge,
                        role_chips_container,
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
