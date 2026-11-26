# SPDX-License-Identifier: Apache-2.0
"""Translate the physical card selector to the packed replay's logical NPU zero."""
import os


def physical_card(*, require_visible=False):
    card = os.environ.get('PEER_PHYSICAL_NPU', '4')
    visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    if card not in ('3', '4') or (require_visible or visible is not None) and card != visible:
        raise ValueError(f'Expected one selected physical card, got physical={card} visible={visible}')
    return card
