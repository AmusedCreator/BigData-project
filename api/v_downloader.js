const fs = require('fs');
const axios = require('axios');
const path = require('path');

//126925366

const DATASET_DIR = '../dataset/vacancies';
const MAX_FILE_SIZE = 500 * 1024 * 1024;
const MAX_RETRIES = 3;
const REQUESTS_PER_SECOND = 10;
const REQUEST_INTERVAL = 2000 / REQUESTS_PER_SECOND;

let currentOutputFile = null;
let processedIDs = new Set();
let lastRequestTime = 0;

function parseArgs() {
  const args = process.argv.slice(2);
  if (args.length < 2) {
    console.error('Использование: node script.js <START_ID> <COUNT>');
    process.exit(1);
  }

  const startId = parseInt(args[0]);
  const count = parseInt(args[1]);

  if (isNaN(startId) || isNaN(count) || startId < 1 || count < 1) {
    console.error('Ошибка: START_ID и COUNT должны быть положительными числами');
    process.exit(1);
  }

  return { startId, count };
}

function findExistingOutputFile(startId) {
  const files = fs.readdirSync(DATASET_DIR).filter(file =>
    file.endsWith('.json') && file.startsWith('hh_vacancies_')
  );

  for (const file of files) {
    const filePath = path.join(DATASET_DIR, file);
    const stats = fs.statSync(filePath);

    if (stats.size < MAX_FILE_SIZE) {
      const match = file.match(/hh_vacancies_(\d+)_to_(\d+)\.json/);
      if (match) {
        const fileStartId = parseInt(match[1]);
        const fileEndId = parseInt(match[2]);
        if (startId <= fileEndId && startId >= fileStartId) {
          return { path: filePath, startId: fileStartId, endId: fileEndId };
        }
      }
    }
  }

  return null;
}

function createNewOutputFile(startId, endId) {
  const fileName = `hh_vacancies_${startId}_to_${endId}.json`;
  const filePath = path.join(DATASET_DIR, fileName);

  // Если файл уже существует, используем его, иначе создаем новый
  if (!fs.existsSync(filePath)) {
    fs.writeFileSync(filePath, '[\n', 'utf8');
  }

  return { path: filePath, startId, endId };
}

function getCurrentOutputFile(startId, endId) {
  const existingFile = findExistingOutputFile(startId);
  if (existingFile) {
    console.log(`Продолжаем запись в существующий файл: ${path.basename(existingFile.path)}`);
    return existingFile;
  }

  const newFile = createNewOutputFile(startId, endId);
  console.log(`Создан новый файл: ${path.basename(newFile.path)}`);
  return newFile;
}

function writeVacancyToFile(vacancy, isFirstInFile = false) {
  const dataToWrite = isFirstInFile
    ? JSON.stringify(vacancy)
    : ',\n' + JSON.stringify(vacancy);

  fs.appendFileSync(currentOutputFile.path, dataToWrite, 'utf8');
}

function finalizeOutputFile() {
  // Проверяем, что файл не пустой и не содержит только "[\n"
  const stats = fs.statSync(currentOutputFile.path);
  if (stats.size > 2) {
    fs.appendFileSync(currentOutputFile.path, '\n]', 'utf8');
  }
}

function isFileFull() {
  try {
    const stats = fs.statSync(currentOutputFile.path);
    return stats.size >= MAX_FILE_SIZE;
  } catch {
    return false;
  }
}

function loadProcessedIDs() {
  const idFile = path.join(DATASET_DIR, 'processed_ids.txt');
  if (fs.existsSync(idFile)) {
    const content = fs.readFileSync(idFile, 'utf8');
    return new Set(content.split('\n').filter(Boolean).map(Number));
  }
  return new Set();
}

function saveProcessedIDs() {
  const idFile = path.join(DATASET_DIR, 'processed_ids.txt');
  const content = Array.from(processedIDs).join('\n');
  fs.writeFileSync(idFile, content, 'utf8');
}

async function rateLimitedRequest(requestFn) {
  const now = Date.now();
  const timeSinceLastRequest = now - lastRequestTime;

  if (timeSinceLastRequest < REQUEST_INTERVAL) {
    await new Promise(resolve => setTimeout(resolve, REQUEST_INTERVAL - timeSinceLastRequest));
  }

  lastRequestTime = Date.now();
  return requestFn();
}

async function fetchVacancy(id, attempt = 1) {
  try {
    const response = await rateLimitedRequest(() =>
      axios.get(`https://api.hh.ru/vacancies/${id}`, {
        headers: { 'User-Agent': 'Mozilla/5.0 HH-DataCollector/1.0' },
        timeout: 15000
      })
    );

    processedIDs.add(id);
    return response.data;
  } catch (err) {
    if (err.response?.status === 404) {
      processedIDs.add(id);
    } else if (err.response?.status === 403) {
      console.log("\nОшибка: 403");
      await new Promise(res => setTimeout(res, 10000));
      return fetchVacancy(id, attempt);
    } else if ((err.code === 'ECONNABORTED' || err.code === 'ETIMEDOUT') && attempt <= MAX_RETRIES) {
      await new Promise(res => setTimeout(res, 3000));
      return fetchVacancy(id, attempt + 1);
    } else {
      processedIDs.add(id);
    }
    return null;
  }
}

function getFileDataState(filePath) {
  try {
    const stats = fs.statSync(filePath);
    if (stats.size <= 2) return 'empty'; // Только "[\n"

    const content = fs.readFileSync(filePath, 'utf8');
    if (content.trim().endsWith(']')) return 'completed';
    return 'in_progress';

  } catch {
    return 'empty';
  }
}

async function main() {
  if (!fs.existsSync(DATASET_DIR)) {
    fs.mkdirSync(DATASET_DIR, { recursive: true });
  }

  const { startId, count } = parseArgs();
  const endId = startId;
  const startIdActual = startId - count + 1;

  console.log(`Начало обработки: ID от ${startId} до ${startIdActual}`);
  console.log(`Всего вакансий для обработки: ${count}`);

  processedIDs = loadProcessedIDs();
  console.log(`Загружено обработанных ID: ${processedIDs.size}`);

  currentOutputFile = getCurrentOutputFile(startIdActual, endId);

  // Определяем состояние файла
  const fileState = getFileDataState(currentOutputFile.path);
  let isFirstWrite = fileState === 'empty';

  if (fileState === 'completed') {
    console.log('Внимание: файл уже завершен. Создаем новый файл.');
    currentOutputFile = createNewOutputFile(startIdActual, endId);
    isFirstWrite = true;
  }

  let processed = 0;
  let collected = 0;
  const startTime = Date.now();

  for (let currentId = startId; currentId >= startIdActual; currentId--) {
    if (processedIDs.has(currentId)) {
      processed++;
      continue;
    }

    if (isFileFull()) {
      finalizeOutputFile();
      currentOutputFile = createNewOutputFile(currentId, endId);
      isFirstWrite = true;
    }

    const vacancy = await fetchVacancy(currentId);

    if (vacancy) {
      writeVacancyToFile(vacancy, isFirstWrite);
      isFirstWrite = false;
      collected++;
    }

    processed++;

    if (processed % 100 === 0) {
      saveProcessedIDs();
    }
    if (processed % 10 === 0) {
      const percent = ((processed / count) * 100).toFixed(1);
      process.stdout.write('\r\x1b[K'); // \x1b[K очищает строку от курсора до конца
      process.stdout.write(`Прогресс: ${percent}% | Обработано: ${processed}/${count} | Собрано: ${collected}`);
    }
  }

  finalizeOutputFile();
  saveProcessedIDs();

  const totalTime = (Date.now() - startTime) / 1000;
  console.log(`Готово! Обработано: ${processed}, собрано: ${collected}, время: ${Math.round(totalTime)}с`);
}

process.on('SIGINT', () => {
  console.log('\nСохранение данных...');
  if (currentOutputFile) finalizeOutputFile();
  saveProcessedIDs();
  process.exit(0);
});

main().catch(console.error);