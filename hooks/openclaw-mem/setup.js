#!/usr/bin/env node
/**
 * OpenClaw-Mem Setup Script
 *
 * Prompts for DeepSeek API key and saves configuration
 */

import readline from 'readline';
import fs from 'fs';
import path from 'path';
import os from 'os';

const CONFIG_DIR = path.join(os.homedir(), '.openclaw-mem');
const CONFIG_FILE = path.join(CONFIG_DIR, 'config.json');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function question(prompt) {
  return new Promise((resolve) => {
    rl.question(prompt, resolve);
  });
}

async function setup() {
  console.log('\n');
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║           🧠 OpenClaw-Mem Setup                            ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  console.log('\n');

  // Ensure config directory exists
  if (!fs.existsSync(CONFIG_DIR)) {
    fs.mkdirSync(CONFIG_DIR, { recursive: true });
  }

  // Load existing config
  let config = {};
  if (fs.existsSync(CONFIG_FILE)) {
    try {
      config = JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf-8'));
    } catch (e) {
      config = {};
    }
  }

  // Check if already configured
  const existingKey = process.env.DEEPSEEK_API_KEY || config.deepseekApiKey;
  if (existingKey) {
    const maskedKey = existingKey.slice(0, 8) + '...' + existingKey.slice(-4);
    console.log(`✓ DeepSeek API Key already configured: ${maskedKey}`);

    const reconfigure = await question('\nDo you want to reconfigure? (y/N): ');
    if (reconfigure.toLowerCase() !== 'y') {
      console.log('\nSetup complete! OpenClaw-Mem is ready to use.\n');
      rl.close();
      return;
    }
  }

  console.log('DeepSeek API is used for AI-powered session summarization.');
  console.log('Get your API key at: https://platform.deepseek.com/\n');
  console.log('(Press Enter to skip if you don\'t have one yet)\n');

  const apiKey = await question('Enter your DeepSeek API Key: ');

  if (apiKey.trim()) {
    config.deepseekApiKey = apiKey.trim();

    // Save to config file
    fs.writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2));
    console.log(`\n✓ API Key saved to ${CONFIG_FILE}`);

    // Suggest adding to shell profile
    console.log('\n📝 To use the API key, add this to your shell profile (~/.bashrc or ~/.zshrc):');
    console.log(`\n   export DEEPSEEK_API_KEY="${apiKey.trim()}"\n`);

    // Ask if user wants to auto-add
    const autoAdd = await question('Add to ~/.zshrc automatically? (y/N): ');
    if (autoAdd.toLowerCase() === 'y') {
      const zshrc = path.join(os.homedir(), '.zshrc');
      const exportLine = `\n# OpenClaw-Mem DeepSeek API\nexport DEEPSEEK_API_KEY="${apiKey.trim()}"\n`;

      try {
        fs.appendFileSync(zshrc, exportLine);
        console.log('\n✓ Added to ~/.zshrc');
        console.log('  Run `source ~/.zshrc` or restart your terminal to apply.\n');
      } catch (e) {
        console.log(`\n⚠ Could not write to ~/.zshrc: ${e.message}`);
        console.log('  Please add the export line manually.\n');
      }
    }
  } else {
    console.log('\n⚠ No API key provided. AI summarization will be disabled.');
    console.log('  You can run `npx openclaw-mem-setup` later to configure.\n');
  }

  // Additional configuration
  console.log('─'.repeat(60));
  console.log('\n📁 Data storage location: ~/.openclaw-mem/');
  console.log('📊 Database: ~/.openclaw-mem/memory.db');
  console.log('🔌 HTTP API port: 18790');
  console.log('\n✓ Setup complete! OpenClaw-Mem is ready to use.\n');

  rl.close();
}

// Run setup
setup().catch(err => {
  console.error('Setup failed:', err.message);
  process.exit(1);
});
