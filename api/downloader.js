//TODO: Что нужно сделать/переделать
//  1) не хранить датасет на гите
//  2) ускорить сборку данных
//  3) собрать больше данных
const fs = require('fs');
const axios = require('axios'); //даунгред до 0.27

const OUTPUT_FILE = '../dataset/hh_vacancies_all.json';
const ID_FILE = '../dataset/hh_vacancies_all_ids.txt';
const TEMP_FILE = '../dataset/hh_vacancies_temp.json';

const BATCH_SIZE = 25;
const MAX_RETRIES = 3;
const MAX_MEMORY_RECORDS = 500;

let processedIDs = [];

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

// Загружаем только обработанные ID
if (fs.existsSync(ID_FILE)) {
  processedIDs = fs.readFileSync(ID_FILE, 'utf8').split('\n').filter(Boolean).map(Number);
}

function saveVacanciesToFile(vacancies) {
  try {
    if (vacancies.length === 0) return;
    
    // Если файл очень большой, используем потоковую запись
    if (fs.existsSync(OUTPUT_FILE) && fs.statSync(OUTPUT_FILE).size > 100 * 1024 * 1024) {
      saveToFileStream(vacancies);
    } else {
      saveToFileDirect(vacancies);
    }
    
    // Сохраняем ID
    fs.writeFileSync(ID_FILE, processedIDs.join('\n'), 'utf8');
    
    console.log(`Сохранили ${vacancies.length} вакансий`);
  } catch (error) {
    console.error('Ошибка при сохранении в файл:', error);
    // Сохраняем во временный файл
    fs.writeFileSync(TEMP_FILE, JSON.stringify(vacancies, null, 2), 'utf8');
    console.log(`Резервная копия сохранена в ${TEMP_FILE}`);
  }
}

function saveToFileDirect(vacancies) {
  let existingVacancies = [];
  
  if (fs.existsSync(OUTPUT_FILE)) {
    const fileContent = fs.readFileSync(OUTPUT_FILE, 'utf8');
    if (fileContent.trim()) {
      existingVacancies = JSON.parse(fileContent);
    }
  }
  
  const allVacancies = [...existingVacancies, ...vacancies];
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(allVacancies, null, 2), 'utf8');
}

function saveToFileStream(vacancies) {
  // Для больших файлов используем потоковое добавление
  const outputStream = fs.createWriteStream(OUTPUT_FILE, { flags: 'a', encoding: 'utf8' });
  
  if (fs.statSync(OUTPUT_FILE).size === 0) {
    // Файл пустой - начинаем массив
    outputStream.write('[\n');
  } else {
    // Файл не пустой - добавляем запятую и новую строку
    outputStream.write(',\n');
  }
  
  // Записываем новые вакансии
  const vacanciesJson = vacancies.map(v => JSON.stringify(v, null, 2)).join(',\n');
  outputStream.write(vacanciesJson);
  
  outputStream.end();
  
  // Нужно закрыть массив в JSON, но это делается при следующем сохранении или в конце
}

function finalizeJsonFile() {
  if (fs.existsSync(OUTPUT_FILE)) {
    const content = fs.readFileSync(OUTPUT_FILE, 'utf8');
    if (!content.trim().endsWith(']')) {
      fs.appendFileSync(OUTPUT_FILE, '\n]');
    }
  }
}

async function fetchVacancy(id, attempt = 1) {
  try {
    const response = await axios.get(`https://api.hh.ru/vacancies/${id}`, {
      headers: { 'User-Agent': 'Mozilla/5.0 HH-DataCollector/1.0' },
      timeout: 15000
    });

    const data = response.data;
    processedIDs.push(id);
    console.log(`Собрана вакансия ID: ${id}`);
    return data;
  } catch (err) {
    if (err.response && err.response.status === 404) {
      processedIDs.push(id);
    } else if (err.response && err.response.status === 403) {
      console.warn(`403 Forbidden на ID ${id}`);
      await new Promise(res => setTimeout(res, 5000));
      return fetchVacancy(id, attempt);
    } else if (err.code === 'ECONNABORTED' || err.code === 'ETIMEDOUT') {
      if (attempt <= MAX_RETRIES) {
        console.warn(`Таймаут на ID ${id}, попытка ${attempt}/${MAX_RETRIES}`);
        await new Promise(res => setTimeout(res, 3000));
        return fetchVacancy(id, attempt + 1);
      } else {
        processedIDs.push(id);
      }
    } else {
      console.error(`Ошибка на ID ${id}: ${err.message}`);
      processedIDs.push(id);
    }
    return null;
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
  
  let currentBatchVacancies = [];
  let totalCollected = 0;

  console.log(`Начало обработки диапазона ID: ${START_ID}-${END_ID}`);
  console.log(`Всего для обработки: ${totalToProcess} ID`);
  console.log(`Максимум в памяти: ${MAX_MEMORY_RECORDS} записей`);

  // Инициализируем файл если его нет
  if (!fs.existsSync(OUTPUT_FILE)) {
    fs.writeFileSync(OUTPUT_FILE, '[\n]', 'utf8');
  }

  for (let i = START_ID; i <= END_ID; i += BATCH_SIZE) {
    const batchPromises = [];
    
    for (let j = i; j < i + BATCH_SIZE && j <= END_ID; j++) {
      if (!processedIDs.includes(j)) {
        batchPromises.push(fetchVacancy(j));
        await new Promise(res => setTimeout(res, 100 + Math.random() * 100));
      }
    }

    const batchResults = await Promise.all(batchPromises);
    const successfulVacancies = batchResults.filter(v => v !== null);
    currentBatchVacancies.push(...successfulVacancies);
    totalCollected += successfulVacancies.length;
    
    processed += batchPromises.length;

    // Сохраняем при достижении лимита памяти
    if (currentBatchVacancies.length >= MAX_MEMORY_RECORDS) {
      console.log(`Достигнут лимит памяти (${currentBatchVacancies.length} записей), сохраняем...`);
      saveVacanciesToFile(currentBatchVacancies);
      currentBatchVacancies = [];
    }

    const percent = ((processed / totalToProcess) * 100).toFixed(2);
    const elapsed = (Date.now() - startTime) / 1000;
    const estimatedTotal = elapsed / processed * totalToProcess;
    const remaining = estimatedTotal - elapsed;

    console.log(`Процесс: ${percent}% — Проанализировано: ${processed}/${totalToProcess} — Собрано: ${totalCollected} — В памяти: ${currentBatchVacancies.length} — Время: ${formatTime(elapsed)} — Осталось: ~${formatTime(remaining)}`);

    await new Promise(res => setTimeout(res, 1000 + Math.random() * 1000));
  }

  // Сохраняем оставшиеся вакансии
  if (currentBatchVacancies.length > 0) {
    console.log(`Сохранение оставшихся ${currentBatchVacancies.length} вакансий...`);
    saveVacanciesToFile(currentBatchVacancies);
  }

  // Финализируем JSON файл
  finalizeJsonFile();
  
  console.log(`\nГотово! Всего собрано вакансий: ${totalCollected}`);
}

// Обработка завершения процесса
process.on('SIGINT', () => {
  console.log('\nПолучен сигнал прерывания, сохраняем данные...');
  finalizeJsonFile();
  process.exit(0);
});

main();