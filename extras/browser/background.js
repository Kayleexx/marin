// Sends {url, title} of the active tab when it changes. No page content, no cookies, no history.
const HOST = "marin_awareness";

function tell(tab) {
  if (!tab || !tab.active || !tab.url || tab.incognito) return;
  browser.runtime.sendNativeMessage(HOST, { url: tab.url, title: tab.title || "" }).catch(() => {});
}

browser.tabs.onActivated.addListener(({ tabId }) => browser.tabs.get(tabId).then(tell));
browser.tabs.onUpdated.addListener((_id, change, tab) => {
  if (change.status === "complete") tell(tab);
});
