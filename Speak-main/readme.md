# Sign Gesture Speak

A real-time sign language recognition web app that converts ASL gestures to text and speech using camera input.

## Prerequisites

- Node.js 18 or newer
- npm 8 or newer

## Install dependencies

```bash
npm install
```

## Run locally

```bash
npm run dev -- --host 0.0.0.0
```

Then open the app in your browser at:

- `http://localhost:5173/`

## Build for production

```bash
npm run build
```

## Preview the production build

```bash
npm run preview
```

## Useful scripts

- `npm run dev` - start the development server
- `npm run build` - build the app for production
- `npm run preview` - preview the production build locally
- `npm run lint` - run ESLint
- `npm run type-check` - run TypeScript type checking

## Notes

The app uses Vite, React, Tailwind CSS, MediaPipe, and TensorFlow.js for real-time gesture detection and speech output.
