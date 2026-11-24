import Artplayer from 'artplayer';
import Hls from 'hls.js';
import artplayerPluginLiveActionSR, { type LiveActionSrPlugin, type Mode } from '../src/index';
import { demoModels } from './models';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const stage = $('stage');
const split = $<HTMLInputElement>('split');
const flip = $<HTMLButtonElement>('flip');
const modeButtons = [...document.querySelectorAll<HTMLButtonElement>('#modes button')];

let art: Artplayer | null = null;
let plugin: LiveActionSrPlugin | null = null;
let mode: Mode = 'performance';
const params = new URLSearchParams(location.search);
// `?model=shipped|test|<url>` and `?ultra=<url>|shipped`: see models.ts.
const models = demoModels(params);
let splitOn = true;
let objectUrl: string | null = null;

function playM3u8(video: HTMLVideoElement, url: string, player: Artplayer): void {
  if (Hls.isSupported()) {
    const hls = new Hls();
    hls.loadSource(url);
    hls.attachMedia(video);
    player.on('destroy', () => hls.destroy());
  } else {
    video.src = url;
  }
}

function load(url: string): void {
  art?.destroy(false);
  art = new Artplayer({
    container: '#player',
    url,
    type: url.includes('.m3u8') ? 'm3u8' : '',
    customType: { m3u8: playM3u8 },
    setting: true,
    fullscreen: true,
    muted: true,
    autoplay: true,
    loop: true,
    moreVideoAttr: { crossOrigin: 'anonymous' },
    plugins: [
      artplayerPluginLiveActionSR({ mode, models, compare: splitOn, onModeChange: showMode }),
    ],
  });
  plugin = art.plugins.artplayerPluginLiveActionSR as LiveActionSrPlugin;
  plugin.setCompare(splitOn, Number(split.value) / 100);
  void plugin.ready.then((ok: boolean) => {
    $('s-gpu').textContent = ok ? 'available' : 'unavailable (plugin stays off)';
  });
  showMode();
}

function setMode(next: Mode): void {
  mode = next;
  plugin?.setMode(next);
  showMode();
}

function showMode(): void {
  for (const button of modeButtons) {
    button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
  }
}

function applySplit(): void {
  plugin?.setCompare(splitOn, Number(split.value) / 100);
  flip.textContent = splitOn ? 'Split on' : 'Split off';
  $('tag-left').style.display = splitOn ? '' : 'none';
}

function openFile(file: File | undefined): void {
  if (!file) return;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  load(objectUrl);
}

const size = (s: { width: number; height: number } | null) => (s ? `${s.width}×${s.height}` : '-');

setInterval(() => {
  if (!plugin) return;
  const stats = plugin.getStats();
  $('s-active').textContent =
    stats.active + (mode === 'auto' ? ' (auto)' : '') + (stats.struggling ? ', struggling' : '');
  $('s-ms').textContent = stats.frameMs === null ? '-' : `${stats.frameMs.toFixed(2)} ms`;
  $('s-native').textContent = size(stats.native);
  $('s-target').textContent = size(stats.target);
  $('s-up').textContent = stats.stages.length ? stats.stages.join(' > ') : '-';
}, 500);

$('file').addEventListener('change', (event) =>
  openFile((event.target as HTMLInputElement).files?.[0]),
);
$('source').addEventListener('submit', (event) => {
  event.preventDefault();
  const url = $<HTMLInputElement>('url').value.trim();
  if (url) load(url);
});
for (const button of modeButtons) {
  button.addEventListener('click', () => setMode(button.dataset.mode as Mode));
}
split.addEventListener('input', applySplit);
flip.addEventListener('click', () => {
  splitOn = !splitOn;
  applySplit();
});

stage.addEventListener('dragover', (event) => {
  event.preventDefault();
  stage.classList.add('drop');
});
stage.addEventListener('dragleave', () => stage.classList.remove('drop'));
stage.addEventListener('drop', (event) => {
  event.preventDefault();
  stage.classList.remove('drop');
  openFile(event.dataTransfer?.files[0]);
});

document.addEventListener('keydown', (event) => {
  if (event.target instanceof HTMLInputElement && event.target.type === 'url') return;
  if (event.key === 's' || event.key === 'S') {
    splitOn = !splitOn;
    applySplit();
  }
  const index = ['1', '2', '3', '4', '5', '6'].indexOf(event.key);
  if (index >= 0) setMode(modeButtons[index].dataset.mode as Mode);
});

const initial = params.get('url');
if (initial) load(initial);
showMode();
