# Solar Power Website

A modern dashboard web application for solar power analytics, built with [Next.js](https://nextjs.org/), [TypeScript](https://www.typescriptlang.org/), [Tailwind CSS](https://tailwindcss.com/), and [shadcn/ui](https://ui.shadcn.com/) components. [Front End Repo Link]([https://ui.shadcn.com/](https://github.com/Ramahabir/solar-power-website))

## Features

- ⚡ **Dashboard UI:** Interactive, responsive, and accessible user interface
- 📊 **Solar Data Visualization:** Real-time charts and analytics for solar power metrics
- 🚀 **Next.js App Router:** Uses Next.js server components and route handlers for data fetching
- 🎨 **Tailwind Styling:** Consistent, utility-first styling for rapid development
- 🧩 **TypeScript:** Strongly typed codebase for reliability and maintainability
- 🔒 **Authentication:** Secure session management and authentication patterns
- 🌍 **Responsive Design:** Mobile-friendly layouts using Tailwind CSS

## Tech Stack

- **Frontend:** Next.js, TypeScript, React, Tailwind CSS, shadcn/ui
- **State Management:** React hooks (local), [Zustand](https://zustand-demo.pmnd.rs/) or [Jotai](https://jotai.org/) for global state (if needed)
- **Data Fetching:** Next.js server components, route handlers
- **Styling:** Tailwind CSS utility classes, CSS Modules (if applicable)
- **Authentication:** Secure patterns recommended for Next.js (e.g., NextAuth.js)

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ramahabir/solar-power-website.git
   cd solar-power-website
   ```

2. **Install dependencies:**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Configure environment variables:**
   - Copy `.env.example` to `.env.local` and fill in the required values.

4. **Run the development server:**
   ```bash
   npm run dev
   # or
   yarn dev
   ```

5. **Open [http://localhost:3000](http://localhost:3000) in your browser.**

## Project Structure

```
├── app/             # Next.js App Router pages & components
├── components/      # Reusable UI components (TypeScript, shadcn/ui)
├── styles/          # Tailwind CSS and global styles
├── lib/             # Utility functions, API clients
├── public/          # Static assets
├── types/           # TypeScript types
├── ...              # Other Next.js folders and configs
```

## Contributing

Feel free to submit issues or pull requests! Please follow the coding standards below:

- Use TypeScript for type safety
- Follow shadcn/ui patterns for component design
- Ensure responsive design and accessibility
- Use Tailwind CSS for all styling
- Implement error handling and loading states in data fetching

## License

This project is licensed under the MIT License.

---

**Created by [Ramahabir](https://github.com/Ramahabir)**
