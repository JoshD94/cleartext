from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[2]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page()
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((root/'outputs/context-model-review.html').as_uri())
    assert page.locator('tbody tr').count()==20
    for width in [1280,390]:
        page.set_viewport_size({'width':width,'height':900})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(root/f'work/context-review-{width}.png'),full_page=True)
    page.locator('summary').first.click()
    assert page.locator('details').first.get_attribute('open') is not None
    assert not errors
    browser.close()
print('Context HTML verified: 20 rows, desktop/mobile, expandable decisions, no JS errors.')
