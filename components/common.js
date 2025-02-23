export function ShowContent({ showContent = false, children }) {
  if (!showContent) return <></>;
  return children;
}
