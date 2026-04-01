# MeetingGenie - Settings Page (stub)
# Full implementation in Phase 5

import flet as ft
from app_state import AppState


def build_settings_page(page: ft.Page, state: AppState) -> ft.Control:
    return ft.Container(
        content=ft.Column(
            [
                ft.Text("Settings", size=20, weight=ft.FontWeight.BOLD),
                ft.Text(
                    "Configuration options will appear here.",
                    size=14,
                    italic=True,
                    color=ft.Colors.ON_SURFACE_VARIANT,
                ),
                ft.Divider(height=16, color=ft.Colors.TRANSPARENT),
                ft.Icon(ft.Icons.SETTINGS, size=64, color=ft.Colors.ON_SURFACE_VARIANT),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            expand=True,
        ),
        padding=20,
        expand=True,
    )
