import { ShowContent } from "@/components/common";

export function FullPageLoading({ text = "" }) {
  return (
    <div
      className={`h-screen w-full flex flex-col items-center justify-center dark:bg-black bg-white`}
    >
      <div className={`loader`} />
      <ShowContent showContent={Boolean(text)}>
        <div
          className={`mt-5 text-center text-lg animate-pulse font-light text-gray-500`}
        >
          {text}
        </div>
      </ShowContent>
    </div>
  );
}

export function FullPageLoadingOverlay({ loading = false, children }) {
  if (!loading) return children;

  return (
    <div
      className={`fixed inset-0 flex items-center justify-center dark:bg-black/50 bg-white/50`}
    >
      <div className={`loader`} />
    </div>
  );
}

export function LoadingSmall({
  color = "dark:text-white text-black",
  margin = "mr-3",
  size = 5,
}) {
  return (
    <div>
      <svg
        className={`${margin} size-${size} animate-spin ${color}`}
        fill="none"
        viewBox="0 0 24 24"
      >
        <circle
          className="stroke-1 opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        />
      </svg>
    </div>
  );
}
