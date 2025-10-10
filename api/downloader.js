//TODO: Что нужно сделать/переделать
//  1) не хранить датасет на гите
//  2) ускорить сборку данных
//  3) собрать больше данных
const fs = require('fs');
const axios = require('axios'); //даунгред до 0.27

const OUTPUT_FILE = '../dataset/hh_vacancies_all.json';
const ID_FILE = '../dataset/hh_vacancies_all_ids.txt';

const BATCH_SIZE = 25;
const MAX_RETRIES = 3;
const SAVE_COUNT = 500;

let allVacancies = [];
let allIDs = [];

const args = process.argv.slice(2);
let START_ID = 1;
let END_ID = 50000;

if (args.length >= 2) {
  START_ID = parseInt(args[0]);
  END_ID = parseInt(args[1]);
  
  if (isNaN(START_ID) || isNaN(END_ID)) {
    console.error('Ошибка: START_ID и END_ID должны быть числами');
    process.exit(1);
  }
  
  if (START_ID > END_ID) {
    console.error('Ошибка: START_ID не может быть больше END_ID');
    process.exit(1);
  }
  
  console.log(`Запуск с ID от ${START_ID} до ${END_ID}`);
} else {
  console.log(`Используются значения по умолчанию: ID от ${START_ID} до ${END_ID}`);
}

if (fs.existsSync(OUTPUT_FILE)) allVacancies = JSON.parse(fs.readFileSync(OUTPUT_FILE, 'utf8'));
if (fs.existsSync(ID_FILE)) allIDs = fs.readFileSync(ID_FILE, 'utf8').split('\n').filter(Boolean).map(Number);

function saveProgress() {
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(allVacancies, null, 2), 'utf8');
  fs.writeFileSync(ID_FILE, allIDs.join('\n'), 'utf8');
  console.log(`Сохранили ${allVacancies.length} вакансий, ID: ${allIDs.length}`);
}

async function fetchVacancy(id, attempt = 1) {
  try {
    const response = await axios.get(`https://api.hh.ru/vacancies/${id}`, {
      headers: { 'User-Agent': 'Mozilla/5.0 HH-DataCollector/1.0' },
      timeout: 15000
    });

    const data = response.data;
    allVacancies.push(data);
    allIDs.push(id);
    console.log(`Собрана вакансия ID: ${id}`);
  } catch (err) {
    if (err.response && err.response.status === 404) {
    } else if (err.response && err.response.status === 403) {
      console.warn(`403 Forbidden на ID ${id}`);
      await new Promise(res => setTimeout(res, 5000));
      return fetchVacancy(id, attempt);
    } else if (err.code === 'ECONNABORTED' || err.code === 'ETIMEDOUT') {
      if (attempt <= MAX_RETRIES) {
        console.warn(`Таймаут на ID ${id}, попытка ${attempt}/${MAX_RETRIES}`);
        await new Promise(res => setTimeout(res, 3000));
        return fetchVacancy(id, attempt + 1);
      }
    } else {
      console.error(`Ошибка на ID ${id}: ${err.message}`);
    }
  }
}

function formatTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return `${h}ч ${m}м ${s}с`;
}

async function main() {
  let processed = 0;
  const totalToProcess = END_ID - START_ID + 1;
  const startTime = Date.now();

  console.log(`Начало обработки диапазона ID: ${START_ID}-${END_ID}`);
  console.log(`Всего для обработки: ${totalToProcess} ID`);

  for (let i = START_ID; i <= END_ID; i += BATCH_SIZE) {
    const batch = [];
    for (let j = i; j < i + BATCH_SIZE && j <= END_ID; j++) {
      if (!allIDs.includes(j)) batch.push(fetchVacancy(j));
      await new Promise(res => setTimeout(res, 100 + Math.random() * 100));
    }

    await Promise.all(batch);
    processed += Math.min(BATCH_SIZE, END_ID - i + 1);

    const percent = ((processed / totalToProcess) * 100).toFixed(2);
    const elapsed = (Date.now() - startTime) / 1000; // секунды
    const estimatedTotal = (elapsed / processed) * totalToProcess;
    const remaining = estimatedTotal - elapsed;

    console.log(`Процесс: ${percent}% — Проанализировано: ${processed}/${totalToProcess} — Собрано вакансий: ${allVacancies.length} — Время: ${formatTime(elapsed)} — Осталось: ~${formatTime(remaining)}`);

    if (i % SAVE_COUNT === 0) saveProgress();
    await new Promise(res => setTimeout(res, 1000 + Math.random() * 1000));
  }

  saveProgress();
  console.log(`\nГотово! Всего собрано вакансий: ${allVacancies.length}`);
}

main();